// Regression test for llama.cpp-ljb9 (commit acdb192d4).
//
// reset_model_weight_entries() used to GGML_ABORT whenever it found an
// entries_ row with in_use_count > 0 at a model-load boundary, on the theory
// that a live lease there could only mean a leaked mem_handle. That premise
// was wrong: llama.cpp supports several llama_model objects loaded at once
// (see tests/test-thread-safety.cpp, which loads one per GPU plus a CPU
// copy), and each live model's tensors hold real weight leases through
// ggml_tensor_extra_gpu::data_handle. acdb192d4 restored preserve-and-
// continue: a live entry now survives the reset instead of aborting the
// process, and only unreferenced entries are reclaimed.
//
// This test targets ONE invariant of that fix: id_to_key_ is keyed by
// identity alone while entries_ is keyed by identity + layout, so ONE id can
// name SEVERAL entries (the same weight staged in more than one physical
// layout, e.g. via S1-PRELOAD). If the reset erased id_to_key_[id]
// unconditionally whenever it dropped any sibling, a still-live sibling of
// the SAME id would be left with no id_to_key_ row -- not a crash, but a
// silent lookup miss for every future resolve of that weight.
// unified_cache::validate() reports exactly this state as "missing
// id_to_key for live entries".
//
// acdb192d4 did NOT fully solve model-load weight ownership: in_use_count
// == 0 means "nobody is resolving this weight right now", not "unowned" --
// leases are transient around each compute, so a live model's currently-idle
// weights read zero between graphs and reset_model_weight_entries() still
// freed them out from under it. That gap is llama.cpp-0qlw, root-caused as
// the cause of the test-thread-safety SEGV and closed by 9f3a2e0f0, which
// added the model-ownership lifetime (owner_mask / live_model_mask) that
// in_use_count had been standing in for.
//
// The three tests below cover that fix (llama.cpp-0qlw Task 14). They assert
// on the reclaim DECISION rather than on teardown counters, because
// "preserved N of N" is also what a no-op prints -- so every preserve
// assertion here is paired with a mutation control that must go RED.
//
// === Why this harness and not tests/test-thread-safety.cpp ===
//
// Task 14 suggested extending test-thread-safety, which loads one model per
// GPU plus a CPU copy. This file is the better instrument for the same claim,
// on three counts, and the third is decisive:
//
//   1. It is affordable. test-thread-safety peaks at 156-180 GB of Shmem for
//      a 19 MB model (measured 2026-08-01; a loop of it drove this host to
//      206 GB and within ~20 s of a global OOM). This test's registration in
//      ggml/src/ggml-sycl/CMakeLists.txt records its measured cost: "free -g
//      flat at 210 GB, peak RSS 317 MB". A proof nobody can afford to re-run
//      is not a regression guard.
//   2. Its exit status is usable. test-thread-safety currently SEGFAULTs for
//      two unrelated reasons (llama.cpp-oze0, llama.cpp-09ep), so a verdict
//      read from it would be absorbed by failures that have nothing to do
//      with weight ownership.
//   3. It can express the mutation control. The control Task 14 requires is
//      "force live_mask to 0 and confirm the entries are then freed". Here
//      that is one public call, set_live_model_mask(0), against an otherwise
//      identical cache -- so the RED and GREEN runs differ in exactly one
//      variable. In test-thread-safety there is no way to reach that state
//      without editing the source, which would make the control a different
//      build rather than a different input.
//
// === What this file does NOT cover ===
//
// Three tickets, three halves of one claim. Each names its own limit so a
// future reader cannot over-read any of them:
//
//   1. That the backend actually CALLS note_model_load_end /
//      release_model_slot at the real model lifecycle boundaries. These tests
//      drive the cache directly, so they prove the decision is right, not that
//      it is REACHED in production. Tracked as llama.cpp-viga.
//
//   2. That in_use_count is TRUTHFUL when the predicate reads it. Everything
//      below is single-threaded against a synthetic cache, so nothing here can
//      produce a raced counter. llama.cpp-c48l establishes that it can be
//      raced: mem_handle::resolve() was declared const while mutating cached_,
//      gen_, leased_entry_ and the chunk-lease fields with no synchronisation,
//      and ggml_backend_sycl_graph_compute releases its mutex before dispatch
//      on 17 of 19 call sites. Two threads in resolve_slow() together both
//      called release_lease() on one lease, driving in_use_count BELOW the
//      live-reference count.
//
//      That is a second, independent route to freeing an in-use entry, and it
//      is worth being precise about the difference: llama.cpp-0qlw is a wrong
//      DECISION about a correct counter (locking cannot fix it), while
//      llama.cpp-c48l is a correct decision about a CORRUPTED counter (locking
//      is exactly what fixes it). The tests below assume the counter is
//      accurate. They are not evidence about the case where it is not.
//
//   3. Any claim beyond live_mask sensitivity for the mutation controls --
//      see the warning above test_live_model_idle_weights_survive_load_boundary.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-reset-model-weight-lease-preserve

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    // Exit 77, not 0. A build without the SYCL backend proves NOTHING about
    // weight ownership in it, and an exit-0 skip reads as a pass -- the
    // llama.cpp-k208 family of clean answers about the wrong thing. 77 is
    // ctest's SKIP_RETURN_CODE; the registration carries the matching
    // SKIP_RETURN_CODE 77.
    fprintf(stderr,
            "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about weight\n"
            "      ownership at a model-load boundary.\n");
    return 77;
}
#else

// Model load is not a quiescent point for OTHER models: reset_model_weight_entries()
// must preserve any entry a live lease still references, and must keep id_to_key_
// pointed at a surviving sibling rather than dangling or silently missing.
static bool test_reset_preserves_leased_entry_and_remaps_id(sycl::queue & q) {
    printf("\n=== Test: reset_model_weight_entries preserves leased layout sibling ===\n");

    ggml_sycl::unified_cache cache(q, 64 * 1024);
    std::vector<uint8_t>     data(128, 0x5a);
    ggml_sycl_cache_id       key = ggml_sycl::test_make_cache_id(data.data());

    // Two layout siblings sharing one id, mirroring how S1-PRELOAD stages the
    // same dense weight in more than one physical layout.
    auto aos = cache.direct_stage_weight(key, data.data(), data.size(), data.size(), GGML_LAYOUT_AOS, nullptr,
                                         nullptr, &q);
    if (!aos.ok || !aos.ptr) {
        fprintf(stderr, "direct_stage_weight failed for AOS layout\n");
        return false;
    }
    aos.event.wait();

    auto soa = cache.direct_stage_weight(key, data.data(), data.size(), data.size(), GGML_LAYOUT_SOA, nullptr,
                                         nullptr, &q);
    if (!soa.ok || !soa.ptr) {
        fprintf(stderr, "direct_stage_weight failed for SOA layout\n");
        return false;
    }
    soa.event.wait();

    if (!cache.is_cached(key, GGML_LAYOUT_AOS) || !cache.is_cached(key, GGML_LAYOUT_SOA)) {
        fprintf(stderr, "Expected both AOS and SOA entries cached before reset\n");
        return false;
    }
    if (!cache.validate()) {
        fprintf(stderr, "Cache validation rejected valid multi-layout entries before reset\n");
        return false;
    }

    // Simulate a DIFFERENT still-loaded model holding a lease on one of the
    // siblings. acquire_weight_lease() tries COALESCED, then SOA, then AOS
    // (unified-cache.cpp), so with only AOS+SOA staged it resolves to SOA.
    auto lease = cache.acquire_weight_lease(key);
    if (!lease || !lease.ptr) {
        fprintf(stderr, "acquire_weight_lease failed to resolve either layout sibling\n");
        return false;
    }
    if (lease.layout != GGML_LAYOUT_SOA) {
        fprintf(stderr, "test assumption violated: acquire_weight_lease did not resolve SOA (got layout=%d)\n",
                (int) lease.layout);
        return false;
    }

    // This is the call under test: the model-load boundary for a DIFFERENT
    // model. Before acdb192d4 this GGML_ABORTed the whole process.
    cache.reset_model_weight_entries();

    // The leased SOA sibling must survive the reset...
    if (!cache.is_cached(key, GGML_LAYOUT_SOA)) {
        fprintf(stderr, "leased SOA sibling was erased by reset_model_weight_entries\n");
        return false;
    }
    // ...the unleased AOS sibling must be reclaimed...
    if (cache.is_cached(key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "unleased AOS sibling was NOT reclaimed by reset_model_weight_entries\n");
        return false;
    }
    // ...and id_to_key_ must still resolve to the surviving SOA entry -- the
    // remap_or_erase_id_mapping_locked() invariant acdb192d4 restores. A flat
    // id_to_key_.erase(id) here would orphan the SOA entry: not a crash, but
    // every future get_weight_ptr(key) / resolve_slow() for this weight
    // would silently miss instead.
    if (!cache.is_cached_any(key)) {
        fprintf(stderr, "id_to_key_ no longer resolves this id after reset (silent lookup miss)\n");
        return false;
    }
    if (!cache.validate()) {
        fprintf(stderr, "validate() reports a stale/dangling id_to_key_ mapping after reset\n");
        return false;
    }

    return true;
}

// ⚠️ Expected noise, not a failure: every unified_cache built after the first
// in this process logs
//
//   [VRAM-ARENA] Insufficient budget for single-chunk arena: 0.0 MB < tail ...
//   [VRAM-ARENA] Failed on device 0, falling back to per-entry allocation
//
// The first cache reserves the arena and later ones see no budget left, so they
// fall back to per-entry allocation. Each test below builds its own cache
// deliberately to keep exact LoadTxnId ownership assertions independent, so
// the fallback is unavoidable here.
//
// It does not affect anything these tests assert. Every assertion is entries_
// membership via is_cached(), and the reclaim decision in
// weight_entry_reclaimable() reads only in_use_count, owner_mask and
// owner_tagged; it never consults location, pool_allocated or pinned, which are
// the fields the fallback changes. Staging still succeeds on the fallback path.
// Recorded so a later reader does not chase it when some other assertion fails.

// Produce a cache entry that is cached but IDLE -- in_use_count == 0 -- which
// is the state llama.cpp-0qlw is entirely about: the steady state of every
// loaded model's weights between graphs, because leases are taken and released
// around each compute.
//
// ⚠️ Getting this right is harder than it looks, and getting it WRONG is what
// made the first version of these tests vacuous. A freshly staged entry is NOT
// idle. direct_stage_weight() calls make_direct_entry_handle(), which bumps
// in_use_count and parks the resulting mem_handle in direct_weight_entries_,
// so the entry reads in_use_count == 1 and the LEASE veto in
// weight_entry_reclaimable() fires FIRST -- preserving the entry before
// ownership is ever consulted. A live_mask mutation against such an entry
// changes nothing, because live_mask is not what is doing the preserving.
//
// The way out, and there is no public API for it: direct_weight_entries_ is
// keyed by IDENTITY ALONE, while entries_ is keyed by identity + layout. So
// staging a SECOND layout under the same id overwrites the registry row,
// destroys the first layout's handle and drops its lease -- leaving the FIRST
// layout's entry cached and idle. These tests therefore assert on the AOS
// sibling; the SOA sibling stays leased, which is also the realistic shape (a
// hot layout in use alongside a cold one).
//
// This is the same displacement the lease-preserve test above already depends
// on to get its AOS sibling reclaimed -- there it is incidental, here it is
// load-bearing, so it is written down. If direct_weight_entries_ ever becomes
// layout-keyed, these tests will start reporting leased=N and preserving for
// the wrong reason. Re-derive the idle state then; do NOT relax the assertions
// to match, because a test that preserves for the wrong reason still passes.
static bool stage_with_idle_aos_sibling(ggml_sycl::unified_cache & cache,
                                        sycl::queue &              q,
                                        const ggml_sycl_cache_id & key,
                                        std::vector<uint8_t> &     data,
                                        const char *               what) {
    auto aos =
        cache.direct_stage_weight(key, data.data(), data.size(), data.size(), GGML_LAYOUT_AOS, nullptr, nullptr, &q);
    if (!aos.ok || !aos.ptr) {
        fprintf(stderr, "direct_stage_weight(AOS) failed for %s\n", what);
        return false;
    }
    aos.event.wait();

    // Displaces the AOS registry row, dropping AOS's lease.
    auto soa =
        cache.direct_stage_weight(key, data.data(), data.size(), data.size(), GGML_LAYOUT_SOA, nullptr, nullptr, &q);
    if (!soa.ok || !soa.ptr) {
        fprintf(stderr, "direct_stage_weight(SOA) failed for %s\n", what);
        return false;
    }
    soa.event.wait();

    if (!cache.is_cached(key, GGML_LAYOUT_AOS) || !cache.is_cached(key, GGML_LAYOUT_SOA)) {
        fprintf(stderr, "%s: expected both layout siblings cached before the reset\n", what);
        return false;
    }
    return true;
}

// THE test for llama.cpp-0qlw: a live model's IDLE weights must survive a
// load boundary, and the proof must be able to fail.
//
// GREEN phase: one model is live and owns the entry, which is idle. The load
// boundary for a DIFFERENT model must not touch it. Before 9f3a2e0f0 the
// predicate was in_use_count alone, so this entry was freed here -- and the
// next resolve of it faulted in mem_handle::resolve_slow().
//
// RED phase (the mutation control Task 14 requires): the same cache, the same
// entry, the same call, with exactly ONE variable changed -- live_model_mask_
// forced to 0. The entry must now be reclaimed. Without this half the GREEN
// assertion proves nothing: a reset that had been turned into a no-op would
// also "preserve" the entry, and would look identical.
//
// The control is not merely a sensitivity check, it reproduces the DEFECT:
// with live_mask == 0, weight_entry_reclaimable() reduces to precisely the
// pre-9f3a2e0f0 predicate (in_use_count == 0 => reclaimable), so the RED
// phase is the old behaviour, executed by the current binary.
//
// ⚠️ Do not over-read the control. It proves this test is SENSITIVE TO
// live_mask -- not that it would catch every possible regression. A change
// that broke the predicate along some other dimension could still pass. What
// the control rules out is exactly one thing, which is the thing that made
// the counters untrustworthy in the first place: that "preserved N of N" is
// also what a no-op prints. A reader who takes it for more than that will
// trust this file past its evidence.
//
// To get the stronger proof -- that the test fails when the FIX is removed
// rather than when its input is changed -- delete the owner_mask veto in
// weight_entry_reclaimable() (unified-cache.cpp, the three lines
// `if ((entry.owner_mask & live_mask) != 0) { return false; }`), rebuild, and
// this test's GREEN phase must fail with the "live model's IDLE weight was
// freed" message. That costs a full SYCL rebuild, which is why it is a
// recipe here rather than a step in the test.
static bool test_live_model_idle_weights_survive_load_boundary(sycl::queue & q) {
    printf("\n=== Test: a live model's IDLE weights survive a load boundary ===\n");

    ggml_sycl::unified_cache cache(q, 64 * 1024);
    std::vector<uint8_t>     data(128, 0x5a);
    ggml_sycl_cache_id       key = ggml_sycl::test_make_cache_id(data.data());

    if (!stage_with_idle_aos_sibling(cache, q, key, data, "the live model's weight")) {
        return false;
    }

    // Model A's exact load transaction completes.
    cache.test_mark_all_entries_touched_by_load(1);
    cache.note_model_load_end(0, 1);
    if (cache.live_model_mask() != 0x1u) {
        fprintf(stderr, "note_model_load_end(0) did not publish slot 0 as live (mask=0x%08x)\n",
                cache.live_model_mask());
        return false;
    }
    if (cache.owner_tagged_entry_count() != 2) {
        fprintf(stderr, "expected both layout siblings owner-tagged after load end, got %zu\n",
                cache.owner_tagged_entry_count());
        return false;
    }

    // --- GREEN: the load boundary for a second model ---
    cache.reset_model_weight_entries(ggml_sycl::weight_reclaim_mode::LOAD_BOUNDARY);

    // The AOS sibling is the idle one (see stage_with_idle_aos_sibling): no
    // lease, so ONLY owner_mask can be preserving it. That is what makes this
    // assertion about the fix rather than about the lease veto.
    if (!cache.is_cached(key, GGML_LAYOUT_AOS)) {
        fprintf(stderr,
                "FAIL: a live model's IDLE weight was freed at another model's load boundary.\n"
                "      This is llama.cpp-0qlw: in_use_count == 0 means 'nobody is resolving\n"
                "      this at this instant', NOT 'unowned'. The next resolve of this weight\n"
                "      is the SIGSEGV in mem_handle::resolve_slow().\n");
        return false;
    }
    if (!cache.is_cached_any(key)) {
        fprintf(stderr, "id_to_key_ no longer resolves the preserved entry (silent lookup miss)\n");
        return false;
    }
    if (!cache.validate()) {
        fprintf(stderr, "validate() rejects the cache after preserving a live model's idle weight\n");
        return false;
    }
    printf("  GREEN: idle weight owned by live slot 0 survived LOAD_BOUNDARY\n");

    // --- RED: the mutation control ---
    // Only live_model_mask_ changes. Everything else -- cache, entry, layout,
    // in_use_count, owner_mask, owner_tagged, the call itself -- is identical.
    cache.set_live_model_mask(0);
    if (cache.live_model_mask() != 0u) {
        fprintf(stderr, "set_live_model_mask(0) did not take effect (mask=0x%08x)\n", cache.live_model_mask());
        return false;
    }

    cache.reset_model_weight_entries(ggml_sycl::weight_reclaim_mode::LOAD_BOUNDARY);

    if (cache.is_cached(key, GGML_LAYOUT_AOS)) {
        fprintf(stderr,
                "FAIL (mutation control): the entry ALSO survived with live_mask == 0, so no\n"
                "      live model owned it and nothing held a lease. The preserve above is\n"
                "      therefore not evidence of anything -- the reclaim path is a no-op on\n"
                "      this input and the GREEN phase would pass against a deleted predicate.\n"
                "      If the teardown line reports leased=1 for this entry, the setup failed\n"
                "      to make it idle -- see stage_with_idle_aos_sibling.\n");
        return false;
    }
    if (!cache.validate()) {
        fprintf(stderr, "validate() rejects the cache after the control reclaimed the entry\n");
        return false;
    }
    printf("  RED (control): with live_mask=0 the same entry WAS reclaimed -- the test can fail\n");

    return true;
}

// The two call sites want OPPOSITE behaviour, and a single predicate cannot
// serve both (llama.cpp-0qlw). This pins the asymmetry, because the tempting
// wrong fix -- "preserve everything if anything is owned or leased" -- is
// invisible to the test above and would turn the replan into a no-op.
//
// MID_LOAD_REPLAN runs during ONE model's own load. The entries it frees are
// the ones that same in-flight load staged moments earlier, which are still
// UNATTRIBUTED because tagging happens at load end. Freeing them is the entire
// point of the call site: its comment records that keeping them and freeing
// them piecemeal "fragments the TLSF arena enough to reject later planned
// experts even when total free bytes are sufficient."
//
// What it must still not touch is another LIVE model's weights. A live
// model's ownership vetoes reclaim in every mode.
static bool test_host_registration_initial_insert_failures(sycl::queue & q) {
    printf("\n=== Test: host registration insertion failures preserve prior aliases ===\n");
    ggml_sycl::unified_cache cache(q, 64 * 1024);
    std::vector<uint8_t>     dense_data(128, 0x41), expert_data(128, 0x42), alias_data(128, 0x43);
    const auto               dense_key     = ggml_sycl::test_make_cache_id(dense_data.data());
    const auto               expert_key    = ggml_sycl::test_make_cache_id(expert_data.data());
    const auto               alias_key     = ggml_sycl::test_make_cache_id(alias_data.data());
    const size_t             baseline_used = cache.used();

    if (!cache.register_host_weight(alias_key, alias_data.data(), alias_data.size(), GGML_LAYOUT_AOS)) {
        return false;
    }
    cache.test_fail_next_host_registration_insert();
    if (cache.register_host_weight(dense_key, dense_data.data(), dense_data.size(), GGML_LAYOUT_AOS) ||
        cache.is_cached(dense_key, GGML_LAYOUT_AOS) || !cache.is_cached(alias_key, GGML_LAYOUT_AOS)) {
        return false;
    }

    if (!cache.register_host_expert(expert_key, expert_data.data(), expert_data.size(), GGML_LAYOUT_AOS)) {
        return false;
    }
    cache.test_fail_next_host_registration_insert();
    if (cache.register_host_expert(dense_key, expert_data.data(), expert_data.size(), GGML_LAYOUT_AOS) ||
        cache.is_cached(dense_key, GGML_LAYOUT_AOS) || !cache.is_cached(expert_key, GGML_LAYOUT_AOS)) {
        return false;
    }

    if (cache.used() != baseline_used) {
        fprintf(stderr, "host registration insertion failure changed cache accounting\n");
        return false;
    }
    return true;
}

static bool test_shared_entry_exact_two_owner_unload(sycl::queue & q) {
    printf("\n=== Test: shared entry retains exact two-model ownership ===\n");

    ggml_sycl::unified_cache cache(q, 64 * 1024);
    std::vector<uint8_t>     data(128, 0x6b);
    ggml_sycl_cache_id       key = ggml_sycl::test_make_cache_id(data.data());
    if (!stage_with_idle_aos_sibling(cache, q, key, data, "shared A/A weight")) {
        return false;
    }

    if (!cache.test_mark_entry_touched_by_load(key, GGML_LAYOUT_AOS, 101)) {
        return false;
    }
    cache.note_model_load_end(0, 101);
    if (!cache.test_mark_entry_touched_by_load(key, GGML_LAYOUT_AOS, 102)) {
        return false;
    }
    cache.note_model_load_end(1, 102);

    cache.release_model_slot(0);
    if (!cache.is_cached(key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "shared entry was freed while its second exact owner remained live\n");
        return false;
    }
    cache.release_model_slot(1);
    if (cache.is_cached(key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "shared entry survived after both exact owners unloaded\n");
        return false;
    }
    return true;
}

static bool test_unrelated_thread_entry_not_claimed(sycl::queue & q) {
    printf("\n=== Test: unrelated-thread insertion is not claimed by loading model ===\n");

    ggml_sycl::unified_cache cache(q, 64 * 1024);
    std::vector<uint8_t>     loading_data(128, 0x31);
    std::vector<uint8_t>     runtime_data(128, 0x32);
    ggml_sycl_cache_id       loading_key = ggml_sycl::test_make_cache_id(loading_data.data());
    ggml_sycl_cache_id       runtime_key = ggml_sycl::test_make_cache_id(runtime_data.data());
    if (!stage_with_idle_aos_sibling(cache, q, loading_key, loading_data, "B staging")) {
        return false;
    }

    bool        thread_ok = false;
    std::thread unrelated([&] {
        thread_ok = stage_with_idle_aos_sibling(cache, q, runtime_key, runtime_data, "unrelated runtime staging");
    });
    unrelated.join();
    if (!thread_ok) {
        return false;
    }

    if (!cache.test_mark_entry_touched_by_load(loading_key, GGML_LAYOUT_AOS, 201)) {
        return false;
    }
    cache.note_model_load_end(0, 201);
    cache.release_model_slot(0);
    if (cache.is_cached(loading_key, GGML_LAYOUT_AOS)) {
        return false;
    }
    if (!cache.is_cached(runtime_key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "unrelated-thread runtime entry was attributed to and freed with model B\n");
        return false;
    }
    return true;
}

static bool test_replan_frees_own_staging_but_spares_a_live_model(sycl::queue & q) {
    printf("\n=== Test: MID_LOAD_REPLAN frees its own staging, spares a live model ===\n");

    ggml_sycl::unified_cache cache(q, 64 * 1024);
    std::vector<uint8_t>     data_live(128, 0x5a);
    std::vector<uint8_t>     data_loading(128, 0xa5);
    ggml_sycl_cache_id       key_live    = ggml_sycl::test_make_cache_id(data_live.data());
    ggml_sycl_cache_id       key_loading = ggml_sycl::test_make_cache_id(data_loading.data());

    // Model A loads and completes -- its entries become owned and tagged.
    if (!stage_with_idle_aos_sibling(cache, q, key_live, data_live, "the live model's weight")) {
        return false;
    }
    cache.test_mark_all_entries_touched_by_load(2);
    cache.note_model_load_end(0, 2);

    // Model B is MID-LOAD: it stages after A's load end, so its entries are
    // untagged. Commit promotes only entries explicitly touched by its exact
    // LoadTxnId; unrelated staging is never claimed.
    if (!stage_with_idle_aos_sibling(cache, q, key_loading, data_loading, "the loading model's staging")) {
        return false;
    }

    cache.reset_model_weight_entries(ggml_sycl::weight_reclaim_mode::MID_LOAD_REPLAN);

    // Both assertions are on the IDLE AOS siblings, so neither outcome can be
    // explained by the lease veto -- ownership is the only thing separating them.
    if (!cache.is_cached(key_live, GGML_LAYOUT_AOS)) {
        fprintf(stderr,
                "FAIL: a replan freed a LIVE model's IDLE weight. Ownership must veto reclaim\n"
                "      in every mode; only UNATTRIBUTED entries are the replan's to free.\n");
        return false;
    }
    if (cache.is_cached(key_loading, GGML_LAYOUT_AOS)) {
        fprintf(stderr,
                "FAIL: the replan did NOT free the in-flight load's own unattributed staging.\n"
                "      That is this call site's entire purpose -- keeping these entries and\n"
                "      freeing them piecemeal is the TLSF fragmentation its comment warns of.\n");
        return false;
    }
    if (!cache.validate()) {
        fprintf(stderr, "validate() rejects the cache after the replan\n");
        return false;
    }
    printf("  live model's weight preserved; in-flight unattributed staging reclaimed\n");

    return true;
}

// The other half of the asymmetry, and the reason the modes cannot be merged.
//
// At a LOAD BOUNDARY an unattributed entry is NOT the caller's own staging --
// it was materialized at runtime, outside any load, so it cannot be attributed
// to a model at all. While ANY model is live it is kept, because there is no
// way to prove nobody can still resolve it. Same entry, same idle state, same
// live model as the replan test above; only the mode differs, and the outcome
// must invert.
static bool test_load_boundary_keeps_unattributed_while_a_model_is_live(sycl::queue & q) {
    printf("\n=== Test: LOAD_BOUNDARY keeps unattributed entries while a model is live ===\n");

    ggml_sycl::unified_cache cache(q, 64 * 1024);
    std::vector<uint8_t>     data_live(128, 0x5a);
    std::vector<uint8_t>     data_runtime(128, 0xa5);
    ggml_sycl_cache_id       key_live    = ggml_sycl::test_make_cache_id(data_live.data());
    ggml_sycl_cache_id       key_runtime = ggml_sycl::test_make_cache_id(data_runtime.data());

    if (!stage_with_idle_aos_sibling(cache, q, key_live, data_live, "the live model's weight")) {
        return false;
    }
    cache.test_mark_all_entries_touched_by_load(3);
    cache.note_model_load_end(0, 3);
    if (!stage_with_idle_aos_sibling(cache, q, key_runtime, data_runtime, "the runtime-materialized entry")) {
        return false;
    }

    cache.reset_model_weight_entries(ggml_sycl::weight_reclaim_mode::LOAD_BOUNDARY);

    if (!cache.is_cached(key_live, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "FAIL: the live model's owned IDLE weight was freed at a load boundary\n");
        return false;
    }
    if (!cache.is_cached(key_runtime, GGML_LAYOUT_AOS)) {
        fprintf(stderr,
                "FAIL: an unattributed entry was freed at a load boundary while a model is\n"
                "      still live. Unattributed means 'cannot be attributed', not 'unowned';\n"
                "      it is reclaimable only once no model is live. Note the same entry IS\n"
                "      correctly freed by MID_LOAD_REPLAN -- that is the mode asymmetry.\n");
        return false;
    }
    if (!cache.validate()) {
        fprintf(stderr, "validate() rejects the cache after the load boundary\n");
        return false;
    }

    // Control: once no model is live, the unattributed entry becomes
    // reclaimable. Without this the assertion above is satisfied by a no-op.
    cache.set_live_model_mask(0);
    cache.reset_model_weight_entries(ggml_sycl::weight_reclaim_mode::LOAD_BOUNDARY);
    if (cache.is_cached(key_runtime, GGML_LAYOUT_AOS)) {
        fprintf(stderr,
                "FAIL (control): the unattributed entry survived with NO model live, so it is\n"
                "      retained unconditionally rather than for the reason claimed.\n");
        return false;
    }
    printf("  unattributed entry kept while a model is live, reclaimed once none is\n");

    return true;
}

// Build a queue on a device already known to exist. Wrapped in a function so
// the queue never has to be default-constructed into a temporary state --
// sycl::queue's default ctor goes through the default selector and throws when
// no device is exposed.
static sycl::queue make_queue_or_fail(const sycl::device & dev) {
    try {
        return sycl::queue(dev);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "FAILED: could not create a SYCL queue on the GPU: %s\n", e.what());
        // A value-returning helper has no natural channel for a failure code,
        // and this is unreachable once get_devices() has reported a non-empty
        // list, so exit(1) buys the same outcome as std::optional unwrapping
        // for less noise. This IS a real failure and stays one -- unlike the
        // no-device case above, which is a skip.
        exit(1);
    }
}

int main() {
    // Enumerate BEFORE constructing any queue. `sycl::queue q;` default-
    // constructs through the default selector, which THROWS when the runtime
    // exposes no device -- and it used to sit outside the try below, so on a
    // device-less host this test died with an uncaught exception (exit 134)
    // without ever reaching its own error handling. Same defect and same fix
    // as test-mmq-xmx-dispatch (0f48c5352, llama.cpp-k208).
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    std::vector<sycl::device> devices;
    try {
        devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL exception while enumerating GPU devices: %s\n", e.what());
        devices.clear();
    }

    if (devices.empty()) {
        // Exit 77, not 0 and not 1. "No device" is not a test result in either
        // direction: 0 is a vacuous pass that reads as verification, 1 is a
        // hard failure on a legitimately CPU-only runner. 77 is ctest's
        // SKIP_RETURN_CODE, and the registration carries the matching
        // SKIP_RETURN_CODE 77 -- without it ctest renders 77 as FAILED.
        fprintf(stderr,
                "SKIP: no SYCL GPU devices available; this run proves NOTHING about weight\n"
                "      ownership at a model-load boundary. Source oneAPI and re-run.\n"
                "      (source /opt/intel/oneapi/setvars.sh --force)\n");
        return 77;
    }

    // devices[0] exists, so this cannot hit the default-selector throw. Note
    // this must NOT be written as `sycl::queue q; ... q = sycl::queue(dev);` --
    // the default construction happens first and throws on a device-less host,
    // which is the very abort the enumeration above exists to avoid.
    sycl::queue q = make_queue_or_fail(devices[0]);
    printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    bool ok = true;
    ok &= test_reset_preserves_leased_entry_and_remaps_id(q);
    ok &= test_live_model_idle_weights_survive_load_boundary(q);
    ok &= test_host_registration_initial_insert_failures(q);
    ok &= test_shared_entry_exact_two_owner_unload(q);
    ok &= test_unrelated_thread_entry_not_claimed(q);
    ok &= test_replan_frees_own_staging_but_spares_a_live_model(q);
    ok &= test_load_boundary_keeps_unattributed_while_a_model_is_live(q);

    printf("\nreset_model_weight_entries lease-preserve tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
#endif
