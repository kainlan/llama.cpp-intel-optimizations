//
// A losing candidate's SCRATCH growth is returned (docs/design/sycl-mmap-host-tier.md §7 C4
// test-sycl-scratch-trial-return; §4.4, §M75; r8 m-1, m-2; r9 m-2, m-3; r10 m-3, m-4; r11 m-4, m-5; r14 M-3;
// r15 I-1; r16 M-3, M-5; r17 M-2; r18 N-6; §M88, §M92).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN: its pinned chunks come from a real queue, so it is registered `sycl;cache;mem-handle` plus gpu-serial,
// serial, with the selector pinned to one discrete card. It loads no model.
//
// The fixture is device 0's cache (the selector's one card): 64 MiB chunks (GGML_SYCL_PINNED_CHUNK_MB=64), SCRATCH
// configured at 64 MiB, and a plan term of 0 for the fixture's notifies, so an ensure's target is the sum of the live
// demands (§7 C4). A notify is the ledger's notify_context on that cache, keyed as the backend's own notify keys it, by
// the context backend's host_zone_owner_id. The transaction steps go through the ggml_backend_sycl_host_scratch_trial
// proc-table entry, as llama_context makes them. A stand-in for a host compute buffer is a SCRATCH allocation.
//
// Every expected figure is a sum of chunk_footprint_for terms (the pool's test_chunk_usable_for and
// test_chunk_backing_for), never a difference, and every figure point also checks that SCRATCH capacity does not exceed
// the pool's allocated(), so a second decrement cannot hide behind a wrapped size_t. Every check of capacity sits
// beside a check of chunk_count or allocated(), because a chunk withdrawn and never released is invisible to capacity.
//
// Cases (each in its own child process; see scratch-return-test-common.hpp):
//   ladder              candidate 512 fits the base; 1024 grows one chunk, holds an 80 MiB stand-in, loses and
//                       RETURNs; the settle's begin releases the chunk.
//   second-context      the same ladder over another context's live 100 MiB: it ends at 192 MiB, not 256.
//   live-refusal        the owner's begin over a live allocation refuses the chunk by name and keeps it.
//   window              the cache-path window: a publish after RETURN grows a new chunk, and the returned chunk is
//                       still released by the winner's begin.
//   empty-return        a reserve's own transaction RETURNs an empty chunk: released at once, the footprint leaving with
//                       the memory.
//   throw-kept          the same throw with a stand-in live: another context's begin leaves the chunk silently.
//   retry-return, retry-keep, trial-retry-return, trial-retry-keep
//                       RETURN_TO_MARK's truncation, both endings, outside and inside a trial.
//   dropped-carrier     a released_chunk dropped, and one consumed by release_chunk_backing, each subtract once.
//   strand              a registry-refused free is stranded, then reconciled once the registry drops the row.
//   teardown            a pool destroyed with a stranded entry reports it.
//   registered-exact    the registry accessor compares the whole release key.
//   nine-strand         nine refused frees strand with no abort.
//   status-order        a live and leased chunk is refused for its live allocation.
//
// Mutants (scratch-test-mutant.hpp), each a ctest of its own with its expected line; the registrations in
// ggml/src/ggml-sycl/CMakeLists.txt list them with the cases each one runs.

#include "../common.hpp"
#include "ggml-backend-impl.h"
#include "scratch-return-test-common.hpp"

#include <chrono>
#include <string>
#include <vector>

using namespace scratch_return_test;

namespace {

using trial_fn = decltype(&ggml_backend_sycl_host_scratch_trial);

constexpr const char * k_strand_line   = "release not yet complete after the owner reset";
constexpr const char * k_teardown_line = "still registered at pool destruction";

// The device's cache, its pool, the trial entry point and the ledger the fixture's notifies reach.
struct rig {
    device_fixture        f;
    trial_fn              trial  = nullptr;
    host_scratch_ledger * ledger = nullptr;
    host_scratch_zone_ops ops{};
};

bool rig_open(checks & c, rig & r) {
    if (!entry_environment_ok(c) || !device_fixture_open(c, r.f)) {
        return false;
    }
    // By proc name, as llama_context resolves it: the private fixtures embed the backend, so the registry is theirs.
    r.trial = reinterpret_cast<trial_fn>(
        ggml_backend_reg_get_proc_address(ggml_backend_sycl_reg(), "ggml_backend_sycl_host_scratch_trial"));
    if (r.trial == nullptr) {
        c.void_case("the proc table has no ggml_backend_sycl_host_scratch_trial entry");
        return false;
    }
    if (!r.f.cache->host_scratch_ledger_ops(0, &r.ledger, &r.ops)) {
        c.void_case("device 0's cache gave no SCRATCH ledger");
        return false;
    }
    // The design's plan term of 0: the ensure's target is the sum of the live demands.
    r.ops.plan_scratch_bytes = 0;
    return true;
}

// One llama_context: its backend and the key its demand and its transaction share.
struct context {
    ggml_backend_t backend = nullptr;
    uint64_t       key     = 0;
};

bool context_open(checks & c, context & ctx) {
    ctx.backend = ggml_backend_sycl_init(0);
    if (ctx.backend == nullptr) {
        c.void_case("a context backend did not initialise");
        return false;
    }
    ctx.key = static_cast<const ggml_backend_sycl_context *>(ctx.backend->context)->host_zone_owner_id;
    return true;
}

void context_close(context & ctx) {
    if (ctx.backend != nullptr) {
        ggml_backend_free(ctx.backend);
        ctx.backend = nullptr;
    }
}

void step(const rig & r, const context & ctx, int op, uint32_t n_ubatch) {
    (void) r.trial(ctx.backend, op, n_ubatch);
}

host_scratch_ensure_result demand(const rig & r, const context & ctx, size_t bytes) {
    return r.ledger->notify_context(ctx.key, bytes, 0, r.ops);
}

// One ensure's grow request (host-zone-plan.cpp ensure_locked_then_grow): the deficit plus one alignment of slack per
// buffer the live demands split into (one each: the fixture's chunk cap is 0).
size_t request_for(size_t deficit, size_t n_buffers) {
    return deficit + n_buffers * pinned_chunk_pool::DEFAULT_ALIGNMENT;
}

struct reading {
    size_t chunks    = 0;
    size_t allocated = 0;
    size_t capacity  = 0;
};

reading read(const rig & r) {
    return { r.f.pool->chunk_count(), r.f.pool->allocated(), r.f.pool->zone_capacity(host_zone_id::SCRATCH) };
}

std::string num(size_t v) {
    return std::to_string(v);
}

// SCRATCH capacity never exceeds the pool's allocated() (r9 m-3).
void expect_sane(checks & c, const rig & r, const char * when) {
    const reading now = read(r);
    c.expect(now.capacity <= now.allocated, "SCRATCH capacity " + mib(now.capacity) +
                                                " MiB exceeds the pool's allocated() " + mib(now.allocated) + " MiB " +
                                                when);
}

// A capacity figure. When it misses and reads exactly the loser's high-water, the line after it says so: the
// keep-the-high-water fault leaves the loser's chunk counted, and the test quotes the figure it predicts.
void expect_capacity(checks &            c,
                     const rig &         r,
                     size_t              want,
                     const std::string & when,
                     const char *        what,
                     size_t              high_water) {
    const size_t got = read(r).capacity;
    if (got == want) {
        return;
    }
    c.fail("SCRATCH capacity " + mib(got) + " MiB " + when + ", expected " + mib(want) + " MiB" + what);
    if (high_water != 0 && got == high_water) {
        fprintf(stderr,
                "  the capacity quotes the loser's high-water: %s MiB, the winner's need plus "
                "chunk_footprint_for of the loser's deficit\n",
                mib(high_water).c_str());
    }
}

void expect_chunks(checks & c, const rig & r, size_t want, const std::string & when, const std::string & cause = "") {
    const size_t got = read(r).chunks;
    c.expect(got == want,
             "chunk_count " + num(got) + " " + when + ", expected " + num(want) + (cause.empty() ? "" : ": " + cause));
}

void expect_allocated(checks &            c,
                      const rig &         r,
                      size_t              want,
                      const std::string & when,
                      const std::string & cause = "") {
    const size_t got = read(r).allocated;
    c.expect(got == want, "allocated() " + mib(got) + " MiB " + when + ", expected " + mib(want) + " MiB" +
                              (cause.empty() ? "" : ": " + cause));
}

// A candidate that grows one chunk and holds a stand-in in it, then loses: OPEN, its notify, the stand-in, RETURN.
// The stand-in must not fit any block that existed before the notify. Returns the chunk's index, SIZE_MAX after a
// VOID.
size_t lose(checks &        c,
            const rig &     r,
            const context & ctx,
            uint32_t        n_ubatch,
            size_t          demand_bytes,
            size_t          stand_in,
            size_t          chunks_before,
            void **         stand_in_ptr) {
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_OPEN, n_ubatch);
    (void) demand(r, ctx, demand_bytes);
    if (read(r).chunks != chunks_before + 1) {
        c.void_case("candidate " + num(n_ubatch) + "'s notify of " + mib(demand_bytes) + " MiB grew " +
                    num(read(r).chunks - chunks_before) + " chunk(s), not 1");
        return SIZE_MAX;
    }
    *stand_in_ptr = r.f.cache->host_zone_alloc(host_zone_id::SCRATCH, stand_in, pinned_chunk_pool::DEFAULT_ALIGNMENT);
    const size_t idx = *stand_in_ptr ? chunk_of(r.f.pool, *stand_in_ptr) : SIZE_MAX;
    if (idx == SIZE_MAX || idx != newest_scratch_chunk(r.f.pool)) {
        c.void_case("the " + mib(stand_in) + " MiB stand-in did not land in the chunk candidate " + num(n_ubatch) +
                    " grew");
        return SIZE_MAX;
    }
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, n_ubatch);
    return idx;
}

// --- the ladder (§7 C4: candidate 512 needs 40 MiB, 1024 needs 150 MiB) ---

int case_ladder(checks & c, const rig & r) {
    context ctx;
    if (!context_open(c, ctx)) {
        return c.status();
    }
    const reading b = read(r);

    // Candidate 512 fits the base: no chunk grows, and it wins.
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_OPEN, 512);
    (void) demand(r, ctx, 40 * MiB);
    if (read(r).chunks != b.chunks) {
        c.void_case("candidate 512's 40 MiB notify grew a chunk: the base is not 64 MiB of SCRATCH");
        return c.status();
    }
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_KEEP, 512);

    // Candidate 1024's deficit is 150 - 64 = 86 MiB: one chunk of chunk_footprint_for(86 MiB plus its slack). The
    // 80 MiB stand-in fits only that chunk. It loses at the host-fallback check, with the stand-in live.
    const size_t loser      = r.f.pool->test_chunk_usable_for(request_for(86 * MiB, 1));
    const size_t high_water = b.capacity + loser;
    const size_t refused    = lines(k_refused_line);
    void *       stand_in   = nullptr;
    const size_t idx        = lose(c, r, ctx, 1024, 150 * MiB, 80 * MiB, b.chunks, &stand_in);
    if (idx == SIZE_MAX) {
        return c.status();
    }
    expect_capacity(c, r, b.capacity, "after RETURN", "", high_water);
    expect_chunks(c, r, b.chunks + 1, "right after RETURN", "a returned chunk with a live allocation stays");
    expect_returned(c, r.f, idx);
    c.expect(lines(k_refused_line) == refused, "RETURN printed a SCRATCH return refused line for live chunk " +
                                                   num(idx) + ": a live returned chunk is kept silently");
    expect_sane(c, r, "after RETURN");

    // The simulated scheduler destruction frees the stand-in; the settle re-reserves 512: its begin, then its reserve.
    r.f.cache->host_zone_free(host_zone_id::SCRATCH, stand_in);
    const size_t released = lines(k_released_line);
    (void) r.f.cache->host_scratch_begin_release(ctx.key);
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_MARK, 512);
    (void) demand(r, ctx, 40 * MiB);
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_KEEP, 512);

    expect_capacity(c, r, b.capacity, "after the ladder", " (winner's need)", high_water);
    expect_chunks(c, r, b.chunks, "after the ladder");
    expect_allocated(c, r, b.allocated, "after the ladder");
    c.expect(lines(k_released_line) > released, "no SCRATCH released line for chunk " + num(idx) + " at the settle");
    expect_sane(c, r, "after the ladder");
    expect_marked_set_quiet(c, r.f, idx, ctx.key);
    context_close(ctx);
    return c.status();
}

// --- the second context (r9 n-7, r10 n-5, r11 n-2) ---

int case_second_context(checks & c, const rig & r) {
    context ctx;
    context other;
    if (!context_open(c, ctx) || !context_open(c, other)) {
        return c.status();
    }
    const reading b0 = read(r);

    // The other context's 100 MiB: deficit 36 MiB, one 64 MiB chunk. It holds its demand as live fillers, so no
    // existing block keeps 58 MiB free.
    (void) demand(r, other, 100 * MiB);
    if (read(r).chunks != b0.chunks + 1) {
        c.void_case("the other context's 100 MiB notify did not grow one chunk");
        return c.status();
    }
    const size_t        other_chunk = r.f.pool->test_chunk_usable_for(request_for(36 * MiB, 1));
    std::vector<void *> other_fill  = scratch_fill_below(r.f.cache, 58 * MiB);
    const reading       b           = read(r);  // the ladder starts here

    // Candidate 512: target 140 MiB, deficit 12 MiB, one 64 MiB chunk K; it wins.
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_OPEN, 512);
    (void) demand(r, ctx, 40 * MiB);
    if (read(r).chunks != b.chunks + 1) {
        c.void_case("candidate 512's notify over the other context did not grow one chunk");
        return c.status();
    }
    const size_t kept = r.f.pool->test_chunk_usable_for(request_for(12 * MiB, 2));
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_KEEP, 512);

    // K is empty once 1024's scheduler replaces 512's, and zone_alloc would visit it first: the K filler takes it.
    void * k_fill = r.f.cache->host_zone_alloc(
        host_zone_id::SCRATCH,
        r.f.cache->host_zone_largest_free_block(host_zone_id::SCRATCH) - pinned_chunk_pool::DEFAULT_ALIGNMENT,
        pinned_chunk_pool::DEFAULT_ALIGNMENT);
    if (k_fill == nullptr || r.f.cache->host_zone_largest_free_block(host_zone_id::SCRATCH) >= 58 * MiB) {
        c.void_case("the K filler left a block of 58 MiB free");
        return c.status();
    }

    // Candidate 1024: target 250 MiB, deficit 58 MiB, one 64 MiB chunk; a 58 MiB stand-in fits only it; it loses.
    const size_t want       = b0.capacity + other_chunk + kept;
    const size_t high_water = want + r.f.pool->test_chunk_usable_for(request_for(58 * MiB, 2));
    void *       stand_in   = nullptr;
    const size_t idx        = lose(c, r, ctx, 1024, 150 * MiB, 58 * MiB, b.chunks + 1, &stand_in);
    if (idx == SIZE_MAX) {
        return c.status();
    }
    expect_capacity(c, r, want, "after RETURN", "", high_water);
    expect_returned(c, r.f, idx);
    expect_sane(c, r, "after RETURN");

    // 1024's scheduler destruction frees the K filler and the stand-in, before the settle's begin.
    r.f.cache->host_zone_free(host_zone_id::SCRATCH, k_fill);
    r.f.cache->host_zone_free(host_zone_id::SCRATCH, stand_in);
    (void) r.f.cache->host_scratch_begin_release(ctx.key);
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_MARK, 512);
    (void) demand(r, ctx, 40 * MiB);
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_KEEP, 512);

    expect_capacity(c, r, want, "after the ladder", " (winner's need)", high_water);
    expect_chunks(c, r, b.chunks + 1, "after the ladder");
    expect_sane(c, r, "after the ladder");

    // The ledger follows the withdrawal: a target above the capacity left after it grows again.
    const size_t target = 160 * MiB + 40 * MiB;
    (void) demand(r, other, 160 * MiB);
    c.expect(read(r).capacity >= target, "SCRATCH capacity " + mib(read(r).capacity) + " MiB after a " + mib(target) +
                                             " MiB target, expected at least " + mib(target) +
                                             " MiB: the ledger still counts the returned chunk");
    scratch_free_all(r.f.cache, other_fill);
    context_close(ctx);
    context_close(other);
    return c.status();
}

// --- a live allocation at the owner's begin ---

int case_live_refusal(checks & c, const rig & r) {
    context ctx;
    if (!context_open(c, ctx)) {
        return c.status();
    }
    const reading b        = read(r);
    void *        stand_in = nullptr;
    const size_t  idx      = lose(c, r, ctx, 1024, 150 * MiB, 80 * MiB, b.chunks, &stand_in);
    if (idx == SIZE_MAX) {
        return c.status();
    }
    const std::string named    = "chunk=" + num(idx) + " reason=live";
    const size_t      refused  = lines(k_refused_line, named);
    const size_t      released = lines(k_released_line);
    (void) r.f.cache->host_scratch_begin_release(ctx.key);
    c.expect(lines(k_refused_line, named) == refused + 1,
             "no SCRATCH return refused line naming " + named + " at its owner's begin over a live allocation");
    c.expect(lines(k_released_line) == released, "a SCRATCH released line printed over live chunk " + num(idx));
    expect_chunks(c, r, b.chunks + 1, "after the owner's begin over a live allocation", "nothing is released under it");
    expect_returned(c, r.f, idx, "after the owner's begin");

    r.f.cache->host_zone_free(host_zone_id::SCRATCH, stand_in);
    (void) r.f.cache->host_scratch_begin_release(ctx.key);
    expect_chunks(c, r, b.chunks, "after the stand-in's free and a begin");
    expect_allocated(c, r, b.allocated, "after the stand-in's free and a begin");
    expect_sane(c, r, "after the release");
    context_close(ctx);
    return c.status();
}

// --- the cache-path window (r8 m-1, r9 m-2) ---

int case_window(checks & c, const rig & r) {
    context ctx;
    if (!context_open(c, ctx)) {
        return c.status();
    }
    const reading b = read(r);

    // A cached candidate 1024 loses and RETURNs its 86 MiB chunk with the 80 MiB stand-in still live.
    const size_t loser    = r.f.pool->test_chunk_usable_for(request_for(86 * MiB, 1));
    void *       stand_in = nullptr;
    const size_t idx      = lose(c, r, ctx, 1024, 150 * MiB, 80 * MiB, b.chunks, &stand_in);
    if (idx == SIZE_MAX) {
        return c.status();
    }
    expect_capacity(c, r, b.capacity, "after RETURN", "", b.capacity + loser);
    expect_returned(c, r.f, idx);

    // The base filled, so no counted chunk has room. Candidate 512's probe and publish make a 1 MiB request through
    // the scheduler's host buffer type: the ensure grows one new chunk and the 1 MiB lands there.
    std::vector<void *>        base_fill = scratch_fill(r.f.cache);
    ggml_backend_buffer_type_t buft      = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(ctx.backend));
    ggml_backend_buffer_t      publish   = buft ? ggml_backend_buft_alloc_buffer(buft, MiB) : nullptr;
    if (publish == nullptr) {
        c.void_case("the host buffer type refused the publish's 1 MiB");
        return c.status();
    }
    const size_t publish_chunk = chunk_of(r.f.pool, ggml_backend_buffer_get_base(publish));
    c.expect(publish_chunk != idx,
             "the publish's 1 MiB landed in returned chunk " + num(idx) + ": a withdrawn chunk takes no allocation");

    // The simulated scheduler destruction frees the stand-in and the filler; candidate 512 OPENs and wins, and only
    // then does its begin run.
    r.f.cache->host_zone_free(host_zone_id::SCRATCH, stand_in);
    scratch_free_all(r.f.cache, base_fill);
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_OPEN, 512);
    (void) demand(r, ctx, 40 * MiB);
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_KEEP, 512);
    const size_t released = lines(k_released_line);
    (void) r.f.cache->host_scratch_begin_release(ctx.key);

    const size_t new_chunk = r.f.pool->test_chunk_usable_for(request_for(MiB, 1));
    const size_t chunks    = read(r).chunks;
    if (chunks != b.chunks + 1) {
        c.fail("chunk_count " + num(chunks) + " after the ladder, expected " + num(b.chunks + 1) +
               (lines(k_released_line) == released ? "; no SCRATCH released line for chunk " + num(idx) : ""));
    } else {
        c.expect(lines(k_released_line) > released, "no SCRATCH released line for chunk " + num(idx));
    }
    c.expect(!r.f.pool->zone_chunk_usage(host_zone_id::SCRATCH, idx).found,
             "returned chunk " + num(idx) + " still in the SCRATCH zone after the winner's begin");
    expect_allocated(c, r, b.allocated + r.f.pool->test_chunk_backing_for(request_for(MiB, 1)), "after the ladder");
    expect_capacity(c, r, b.capacity + new_chunk, "after the ladder", "", 0);
    expect_sane(c, r, "after the ladder");
    expect_marked_set_quiet(c, r.f, idx, ctx.key);
    ggml_backend_buffer_free(publish);
    context_close(ctx);
    return c.status();
}

// --- the empty return (r10 m-3) and the footprint's free point (r14 M-3, r15 I-1) ---

struct free_point {
    size_t fires     = 0;
    size_t allocated = 0;
};

void free_point_sample(void * ctx, const pinned_chunk_pool & pool) {
    auto * s = static_cast<free_point *>(ctx);
    s->fires++;
    s->allocated = pool.allocated();
}

int case_empty_return(checks & c, const rig & r) {
    context ctx;
    if (!context_open(c, ctx)) {
        return c.status();
    }
    const reading b = read(r);

    // Outside any trial, the reserve's own transaction grows one chunk (deficit 36 MiB), and the reserve throws before
    // any host buffer lands in it.
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_MARK, 512);
    (void) demand(r, ctx, 100 * MiB);
    if (read(r).chunks != b.chunks + 1) {
        c.void_case("the reserve's 100 MiB notify did not grow one chunk");
        return c.status();
    }
    const size_t idx     = newest_scratch_chunk(r.f.pool);
    const size_t backing = r.f.pool->test_chunk_backing_for(request_for(36 * MiB, 1));
    free_point   sample;
    r.f.pool->test_set_free_point_hook(free_point_sample, &sample);
    const size_t released = lines(k_released_line);
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 512);
    r.f.pool->test_set_free_point_hook(nullptr, nullptr);

    expect_chunks(c, r, b.chunks, "right after RETURN", "empty returned chunk " + num(idx) + " not released");
    expect_allocated(c, r, b.allocated, "right after RETURN");
    c.expect(lines(k_released_line) > released, "no SCRATCH released line for chunk " + num(idx) + " at RETURN");
    c.expect(sample.fires == 1,
             "the free point fired " + num(sample.fires) + " time(s) during the empty return's release, expected 1");
    if (sample.fires == 1) {
        c.expect(sample.allocated == b.allocated + backing,
                 "allocated() " + mib(sample.allocated) + " MiB at the free point, expected " +
                     mib(b.allocated + backing) + " MiB: footprint decremented before the free");
    }
    expect_sane(c, r, "after the empty return");
    expect_marked_set_quiet(c, r.f, idx, ctx.key);
    context_close(ctx);
    return c.status();
}

// --- the runtime throw with the context kept (r10 m-3) ---

int case_throw_kept(checks & c, const rig & r) {
    context ctx;
    context second;
    if (!context_open(c, ctx) || !context_open(c, second)) {
        return c.status();
    }
    std::vector<void *> base_fill = scratch_fill(r.f.cache);
    const reading       b         = read(r);

    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_MARK, 512);
    (void) demand(r, ctx, 100 * MiB);
    if (read(r).chunks != b.chunks + 1) {
        c.void_case("the reserve's 100 MiB notify did not grow one chunk");
        return c.status();
    }
    void * stand_in = r.f.cache->host_zone_alloc(host_zone_id::SCRATCH, 48 * MiB, pinned_chunk_pool::DEFAULT_ALIGNMENT);
    const size_t idx = stand_in ? chunk_of(r.f.pool, stand_in) : SIZE_MAX;
    if (idx == SIZE_MAX || idx != newest_scratch_chunk(r.f.pool)) {
        c.void_case("the 48 MiB stand-in did not land in the grown chunk");
        return c.status();
    }
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 512);
    expect_returned(c, r.f, idx);

    // The context is kept and no reserve follows. Another context's begin on the device leaves the chunk silently.
    const size_t refused = lines(k_refused_line);
    (void) r.f.cache->host_scratch_begin_release(second.key);
    c.expect(lines(k_refused_line) == refused,
             "the second context's begin printed a SCRATCH return refused line for chunk " + num(idx) +
                 ", another owner's live chunk");
    expect_returned(c, r.f, idx, "after the second context's begin");
    expect_capacity(c, r, b.capacity, "after the second context's begin", "", 0);
    expect_chunks(c, r, b.chunks + 1, "after the second context's begin");
    expect_sane(c, r, "after the second context's begin");

    r.f.cache->host_zone_free(host_zone_id::SCRATCH, stand_in);
    (void) r.f.cache->host_scratch_begin_release(ctx.key);
    expect_chunks(c, r, b.chunks, "after the stand-in's free and the owner's begin");
    scratch_free_all(r.f.cache, base_fill);
    context_close(ctx);
    context_close(second);
    return c.status();
}

// --- the retry's truncation (r10 m-4, r11 m-4) ---

int case_retry(checks & c, const rig & r, bool in_trial, bool keep) {
    context ctx;
    if (!context_open(c, ctx)) {
        return c.status();
    }
    const reading b = read(r);  // before MARK

    if (in_trial) {
        step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_OPEN, 1024);
    }
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_MARK, 1024);
    (void) demand(r, ctx, 150 * MiB);
    if (read(r).chunks != b.chunks + 1) {
        c.void_case("the reserve's 150 MiB notify did not grow one chunk");
        return c.status();
    }
    const size_t grown = r.f.pool->test_chunk_usable_for(request_for(86 * MiB, 1));
    void * stand_in = r.f.cache->host_zone_alloc(host_zone_id::SCRATCH, 80 * MiB, pinned_chunk_pool::DEFAULT_ALIGNMENT);
    const size_t a  = stand_in ? chunk_of(r.f.pool, stand_in) : SIZE_MAX;
    if (a == SIZE_MAX || a != newest_scratch_chunk(r.f.pool)) {
        c.void_case("the 80 MiB stand-in did not land in chunk A");
        return c.status();
    }

    // The pp retry: RETURN_TO_MARK files A and leaves the transaction open with nothing in it.
    step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN_TO_MARK, 1024);
    const size_t held = r.f.cache->host_scratch_trial_chunks(ctx.key);
    c.expect(held == 0, "transaction holds " + num(held) + " chunk(s) after RETURN_TO_MARK, expected 0");
    expect_returned(c, r.f, a, "after RETURN_TO_MARK");
    expect_capacity(c, r, b.capacity, "after RETURN_TO_MARK", "", 0);

    // The retried reserve grows one more chunk, B.
    (void) demand(r, ctx, 150 * MiB);
    const size_t retried = read(r).chunks;
    if (retried != b.chunks + 2) {
        c.fail("the retried reserve after RETURN_TO_MARK grew " +
               num(retried > b.chunks + 1 ? retried - b.chunks - 1 : 0) +
               " chunk(s), expected 1: the returned chunk still counts");
        return c.status();
    }

    if (!keep) {
        step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 1024);
        expect_returned(c, r.f, a);
        expect_capacity(c, r, b.capacity, "after RETURN", "", 0);
        expect_sane(c, r, "after RETURN");
        r.f.cache->host_zone_free(host_zone_id::SCRATCH, stand_in);
        (void) r.f.cache->host_scratch_begin_release(ctx.key);
        expect_capacity(c, r, b.capacity, "after the begin", "", 0);
        expect_chunks(c, r, b.chunks, "after the begin");
        expect_allocated(c, r, b.allocated, "after the begin");
    } else {
        step(r, ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_KEEP, 1024);
        c.expect(r.f.pool->zone_chunk_usage(host_zone_id::SCRATCH, a).withdrawn,
                 "chunk " + num(a) + " not withdrawn after KEEP: KEEP reaches only the transaction's chunks");
        expect_capacity(c, r, b.capacity + grown, "after KEEP", "", 0);
        r.f.cache->host_zone_free(host_zone_id::SCRATCH, stand_in);
        (void) r.f.cache->host_scratch_begin_release(ctx.key);
        expect_chunks(c, r, b.chunks + 1, "after the begin");
        expect_capacity(c, r, b.capacity + grown, "after the begin", "", 0);
        expect_allocated(c, r, b.allocated + r.f.pool->test_chunk_backing_for(request_for(86 * MiB, 1)),
                         "after the begin");
    }
    expect_sane(c, r, "after the begin");
    expect_marked_set_quiet(c, r.f, a, ctx.key);
    context_close(ctx);
    return c.status();
}

// --- a dropped carrier and a consumed one (r15 I-1) ---

// A SCRATCH chunk grown directly on the pool and withdrawn, as RETURN leaves an empty one. SIZE_MAX after a VOID.
size_t withdrawn_chunk(checks & c, const rig & r) {
    size_t idx = pinned_chunk_pool::NO_CHUNK;
    if (!r.f.pool->grow_zone(host_zone_id::SCRATCH, 1, &idx) || idx == pinned_chunk_pool::NO_CHUNK ||
        !r.f.pool->withdraw_zone_chunk(host_zone_id::SCRATCH, idx)) {
        c.void_case("no SCRATCH chunk could be grown and withdrawn");
        return SIZE_MAX;
    }
    return idx;
}

int case_dropped_carrier(checks & c, const rig & r) {
    const reading b       = read(r);
    const size_t  backing = r.f.pool->test_chunk_backing_for(1);

    const size_t dropped = withdrawn_chunk(c, r);
    if (dropped == SIZE_MAX) {
        return c.status();
    }
    free_point sample;
    r.f.pool->test_set_free_point_hook(free_point_sample, &sample);
    {
        pinned_chunk_pool::released_chunk carrier;
        if (r.f.pool->release_zone_chunk(host_zone_id::SCRATCH, dropped, carrier) != PINNED_CHUNK_RELEASE_DONE) {
            r.f.pool->test_set_free_point_hook(nullptr, nullptr);
            c.void_case("the empty withdrawn chunk was not released");
            return c.status();
        }
        // Dropped here, as an early return or a failed append in a caller would drop it.
    }
    r.f.pool->test_set_free_point_hook(nullptr, nullptr);
    c.expect(sample.fires == 1,
             "the free point fired " + num(sample.fires) + " time(s) for a dropped released_chunk, expected 1");
    expect_allocated(
        c, r, b.allocated, "after a dropped released_chunk",
        read(r).allocated > b.allocated ? "footprint leaked by a dropped carrier" : "footprint subtracted twice");

    const size_t consumed = withdrawn_chunk(c, r);
    if (consumed == SIZE_MAX) {
        return c.status();
    }
    expect_allocated(c, r, b.allocated + backing, "with the second chunk grown");
    {
        pinned_chunk_pool::released_chunk carrier;
        if (r.f.pool->release_zone_chunk(host_zone_id::SCRATCH, consumed, carrier) != PINNED_CHUNK_RELEASE_DONE) {
            c.void_case("the second empty withdrawn chunk was not released");
            return c.status();
        }
        pinned_chunk_pool::release_chunk_backing(std::move(carrier));
        // The moved-from carrier leaves scope here: it owns nothing.
    }
    expect_allocated(
        c, r, b.allocated, "after release_chunk_backing",
        read(r).allocated < b.allocated ? "footprint subtracted twice" : "footprint leaked by a consumed carrier");
    expect_sane(c, r, "after the two carriers");
    return c.status();
}

// --- a registry-refused free (§M88 (1), §M92; r16 M-5, M-6) ---

bool strict_leases_set(checks & c) {
    // The registry's own reading, asked of its one accessor.
    if (ggml_sycl_strict_enabled()) {
        c.void_case("GGML_SYCL_STRICT_LEASES is set: the registry's strict classification could abort the held row");
        return true;
    }
    return false;
}

bool chunk_hold(checks & c, const pinned_chunk_pool * pool, size_t idx, alloc_metadata * md) {
    if (!pool->test_chunk_metadata(idx, md) || !allocation_registry_test_hold_physical_release(*md)) {
        c.void_case("the registry row of chunk " + num(idx) + " could not be held");
        return false;
    }
    return true;
}

void release_now(pinned_chunk_pool * pool, size_t idx) {
    pinned_chunk_pool::released_chunk carrier;
    if (pool->release_zone_chunk(host_zone_id::SCRATCH, idx, carrier) == PINNED_CHUNK_RELEASE_DONE) {
        pinned_chunk_pool::release_chunk_backing(std::move(carrier));
    }
}

bool retries_drained(checks & c, int device) {
    const auto coordinator = unified_allocation_release_coordinator(device);
    if (!coordinator) {
        c.fail("no release coordinator for device " + std::to_string(device));
        return false;
    }
    (void) coordinator->process_retries();
    c.expect(coordinator->retry_count() == 0, "the release coordinator still holds " + num(coordinator->retry_count()) +
                                                  " retry(ies) after process_retries");
    return true;
}

int case_strand(checks & c, const rig & r) {
    if (strict_leases_set(c)) {
        return c.status();
    }
    const reading  b       = read(r);
    const size_t   backing = r.f.pool->test_chunk_backing_for(1);
    const size_t   idx     = withdrawn_chunk(c, r);
    alloc_metadata md{};
    if (idx == SIZE_MAX || !chunk_hold(c, r.f.pool, idx, &md)) {
        return c.status();
    }
    const size_t held    = b.allocated + backing;
    const size_t strands = lines(k_strand_line);
    release_now(r.f.pool, idx);

    // The registry refused the physical release: the row is LIVE again, so the footprint stays.
    const size_t first = r.f.pool->allocated();
    if (first != held) {
        if (lines(k_strand_line) == strands) {
            c.fail("allocated() " + mib(first) + " MiB after a refused free, expected " + mib(held) +
                   " MiB: footprint left while the allocation is still registered");
        } else {
            c.fail("allocated() " + mib(first) +
                   " MiB at the first reconcile while the allocation is still registered, expected " + mib(held) +
                   " MiB");
        }
    }
    c.expect(r.f.pool->stranded_count() == 1,
             "stranded_ holds " + num(r.f.pool->stranded_count()) + " entries after a refused free, expected 1");
    c.expect(lines(k_strand_line) == strands + 1, "the strand line printed " + num(lines(k_strand_line) - strands) +
                                                      " time(s) after a refused free, expected 1");
    const size_t second = r.f.pool->allocated();
    c.expect(second == first, "a second allocated() read " + mib(second) + " MiB, the first " + mib(first) + " MiB");
    c.expect(lines(k_strand_line) == strands + 1, "a second allocated() printed the strand line again");

    (void) allocation_registry_test_unhold_physical_release(md);
    if (!retries_drained(c, md.device)) {
        return c.status();
    }
    expect_allocated(c, r, b.allocated, "after the retry completed", "stranded footprint never reconciled");
    c.expect(r.f.pool->stranded_count() == 0,
             "stranded_ holds " + num(r.f.pool->stranded_count()) + " entries after the retry completed, expected 0");
    return c.status();
}

int case_teardown(checks & c, const rig & r) {
    if (strict_leases_set(c)) {
        return c.status();
    }
    alloc_metadata md{};
    size_t         teardowns = 0;
    {
        // A pool of its own on the cache's queue, so the case can destroy it.
        pinned_chunk_pool pool(r.f.cache->get_queue(), 4 * k_chunk_mib * MiB);
        size_t            idx = pinned_chunk_pool::NO_CHUNK;
        if (!pool.configure_zones(MiB, MiB, MiB, k_chunk_mib * MiB) || !pool.zones_configured() ||
            !pool.grow_zone(host_zone_id::SCRATCH, 1, &idx) || idx == pinned_chunk_pool::NO_CHUNK ||
            !pool.withdraw_zone_chunk(host_zone_id::SCRATCH, idx)) {
            c.void_case("the case's own pool could not grow and withdraw a SCRATCH chunk");
            return c.status();
        }
        if (!chunk_hold(c, &pool, idx, &md)) {
            return c.status();
        }
        release_now(&pool, idx);
        c.expect(pool.stranded_count() == 1, "stranded_ holds " + num(pool.stranded_count()) +
                                                 " entries before the pool's destruction, expected 1");
        teardowns = lines(k_teardown_line);
    }
    c.expect(lines(k_teardown_line) == teardowns + 1, "no teardown line for the stranded entry");
    (void) allocation_registry_test_unhold_physical_release(md);
    (void) retries_drained(c, md.device);
    return c.status();
}

// --- the accessor compares the key (r16 M-3, r17 N-5) ---

int case_registered_exact(checks & c, const rig & r) {
    (void) r;
    const size_t   rows = allocation_registry_test_size();
    alloc_metadata row;
    row.ptr        = reinterpret_cast<void *>(static_cast<uintptr_t>(0x5c7a0000));
    row.size       = 256;
    row.device     = 0;
    row.id         = 4242;
    row.generation = 7;
    row.offset     = 64;
    row.extent     = 512;
    row.epoch_id   = 1;
    row.kind       = alloc_tier::HOST_PINNED;
    row.role       = alloc_role::STAGING;
    row.category   = runtime_category::STAGING;
    row.zone       = vram_zone_id::RUNTIME;
    row.host_zone  = host_zone_id::STAGING;
    if (!allocation_registry_test_publish(row, true)) {
        c.void_case("the test row could not be published");
        return c.status();
    }
    const size_t published = allocation_registry_test_size();

    c.expect(unified_allocation_registered_exact(row) == ALLOCATION_REGISTERED_LIVE,
             "registered_exact did not answer LIVE for the published row");
    alloc_metadata other_id = row;
    other_id.id             = row.id + 1;
    if (unified_allocation_registered_exact(other_id) != ALLOCATION_REGISTERED_GONE) {
        char p[32];
        snprintf(p, sizeof(p), "%p", row.ptr);
        c.fail(std::string("registered_exact answered true for ptr ") + p + " with id " + num(other_id.id) +
               ", the row holds id " + num(row.id));
    }
    alloc_metadata absent = row;
    absent.ptr            = static_cast<char *>(row.ptr) + 0x100;
    c.expect(unified_allocation_registered_exact(absent) == ALLOCATION_REGISTERED_GONE,
             "registered_exact answered true for an absent row");
    c.expect(allocation_registry_test_size() == published, "the three queries changed the registry's row count");
    allocation_registry_test_erase(row.ptr);
    c.expect(allocation_registry_test_size() == rows, "the registry was not left as the case found it");
    return c.status();
}

// --- more than eight chunks strand (r17 M-2) ---

int case_nine_strand(checks & c, const rig & r) {
    if (strict_leases_set(c)) {
        return c.status();
    }
    const reading               b = read(r);
    std::vector<size_t>         idxs;
    std::vector<alloc_metadata> mds;
    for (int i = 0; i < 9; ++i) {
        const size_t   idx = withdrawn_chunk(c, r);
        alloc_metadata md{};
        if (idx == SIZE_MAX || !chunk_hold(c, r.f.pool, idx, &md)) {
            for (const alloc_metadata & held : mds) {
                (void) allocation_registry_test_unhold_physical_release(held);
            }
            return c.status();
        }
        idxs.push_back(idx);
        mds.push_back(md);
    }
    for (const size_t idx : idxs) {
        release_now(r.f.pool, idx);
    }
    c.expect(r.f.pool->stranded_count() == 9,
             "stranded_ holds " + num(r.f.pool->stranded_count()) + " entries after nine refused frees, expected 9");
    for (const alloc_metadata & md : mds) {
        (void) allocation_registry_test_unhold_physical_release(md);
    }
    if (!retries_drained(c, mds.front().device)) {
        return c.status();
    }
    expect_allocated(c, r, b.allocated, "after the nine retries completed");
    return c.status();
}

// --- the status order (r18 N-6) ---

int case_status_order(checks & c, const rig & r) {
    context ctx;
    if (!context_open(c, ctx)) {
        return c.status();
    }
    const reading b        = read(r);
    void *        stand_in = nullptr;
    const size_t  idx      = lose(c, r, ctx, 1024, 150 * MiB, 80 * MiB, b.chunks, &stand_in);
    if (idx == SIZE_MAX) {
        return c.status();
    }
    const uint64_t lease = r.f.cache->host_acquire_chunk_lease(stand_in);
    if (lease == pinned_chunk_pool::INVALID_CHUNK_HANDLE) {
        c.void_case("no lease could be taken on chunk " + num(idx));
        return c.status();
    }
    const std::string live   = "chunk=" + num(idx) + " reason=live";
    const std::string leased = "chunk=" + num(idx) + " reason=lease";
    const size_t      before = lines(k_refused_line, live);
    (void) r.f.cache->host_scratch_begin_release(ctx.key);
    c.expect(lines(k_refused_line, live) == before + 1,
             "no SCRATCH return refused line naming " + live + " for a live and leased chunk");
    c.expect(lines(k_refused_line, leased) == 0, "reason=lease for chunk " + num(idx) +
                                                     ", which holds a live allocation: a live allocation is checked "
                                                     "before a lease");

    r.f.cache->host_release_chunk_lease(lease);
    r.f.cache->host_zone_free(host_zone_id::SCRATCH, stand_in);
    (void) r.f.cache->host_scratch_begin_release(ctx.key);
    expect_chunks(c, r, b.chunks, "after the lease's drop, the stand-in's free and a begin");
    context_close(ctx);
    return c.status();
}

int run_case(const child_args & a) {
    checks c{ a.case_name.c_str() };
    if (!child_begin(a)) {
        return 2;
    }
    rig r;
    if (!rig_open(c, r)) {
        return c.status();
    }
    const std::string & n  = a.case_name;
    int                 rc = 2;
    if (n == "ladder") {
        rc = case_ladder(c, r);
    } else if (n == "second-context") {
        rc = case_second_context(c, r);
    } else if (n == "live-refusal") {
        rc = case_live_refusal(c, r);
    } else if (n == "window") {
        rc = case_window(c, r);
    } else if (n == "empty-return") {
        rc = case_empty_return(c, r);
    } else if (n == "throw-kept") {
        rc = case_throw_kept(c, r);
    } else if (n == "retry-return") {
        rc = case_retry(c, r, false, false);
    } else if (n == "retry-keep") {
        rc = case_retry(c, r, false, true);
    } else if (n == "trial-retry-return") {
        rc = case_retry(c, r, true, false);
    } else if (n == "trial-retry-keep") {
        rc = case_retry(c, r, true, true);
    } else if (n == "dropped-carrier") {
        rc = case_dropped_carrier(c, r);
    } else if (n == "strand") {
        rc = case_strand(c, r);
    } else if (n == "teardown") {
        rc = case_teardown(c, r);
    } else if (n == "registered-exact") {
        rc = case_registered_exact(c, r);
    } else if (n == "nine-strand") {
        rc = case_nine_strand(c, r);
    } else if (n == "status-order") {
        rc = case_status_order(c, r);
    } else {
        fprintf(stderr, "unknown case '%s'\n", n.c_str());
    }
    ggml_backend_free(r.f.backend);
    return rc;
}

}  // namespace

int main(int argc, char ** argv) {
    return cases_main(argc, argv,
                      { "ladder", "second-context", "live-refusal", "window", "empty-return", "throw-kept",
                        "retry-return", "retry-keep", "trial-retry-return", "trial-retry-keep", "dropped-carrier",
                        "strand", "teardown", "registered-exact", "nine-strand", "status-order" },
                      run_case, std::chrono::milliseconds(120000));
}
