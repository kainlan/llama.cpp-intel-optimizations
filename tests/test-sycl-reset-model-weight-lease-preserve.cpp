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
// acdb192d4 does NOT fully solve model-load weight ownership: in_use_count
// == 0 means "nobody is resolving this weight right now", not "unowned" --
// leases are transient around each compute, so a live model's currently-idle
// weights read zero between graphs and reset_model_weight_entries() still
// frees them out from under it. That gap is llama.cpp-0qlw (root-caused as
// the cause of the test-thread-safety SEGV) and is untouched by this test.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-sycl-reset-model-weight-lease-preserve

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
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

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }

    sycl::queue q;
    try {
        printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    bool ok = true;
    ok &= test_reset_preserves_leased_entry_and_remaps_id(q);

    printf("\nreset_model_weight_entries lease-preserve tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
#endif
