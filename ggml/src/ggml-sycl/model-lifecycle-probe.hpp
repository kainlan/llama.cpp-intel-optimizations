#pragma once

// Model-lifecycle invocation probe (llama.cpp-viga).
//
// tests/test-sycl-reset-model-weight-lease-preserve.cpp proves the weight-lease
// reclaim DECISION is correct by driving unified_cache directly.  Nothing proved
// the backend ever REACHES those calls on a real llama_model load/teardown, so
// the ownership logic could be perfect and simply never invoked in production
// and the whole suite would stay green.  These counters close that half.
//
// Two properties this deliberately keeps:
//
//   * They are ALWAYS compiled in -- no GGML_SYCL_TEST / NDEBUG gate.  A counter
//     visible only in a test build makes the observed binary a different binary
//     from the shipped one, which is the "control that changes two things at
//     once" trap.  The cost is two relaxed atomic increments per model lifetime.
//
//   * Each counter is bumped AFTER its unified-cache call returns, on a
//     straight-line path.  A bump therefore means the call EXECUTED, not that
//     control merely arrived somewhere nearby.  Entering
//     ggml_backend_sycl_model_unloaded() with a slot it rejects must NOT bump
//     anything -- tests/test-sycl-model-lifecycle-hooks.cpp asserts exactly that.
//
// Deliberately NOT in ggml/include/ggml-sycl.h: that header reaches the whole
// SYCL backend (via common.hpp) and all of llama, and this is a diagnostic, not
// public API.  This header pulls in no SYCL headers, so a plain C++ test TU can
// include it without -fsycl.

#include "ggml-backend.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend_sycl_model_lifecycle_probe {
    // Times unified_cache_note_model_load_end() returned, i.e. the number of
    // model loads whose completion the cache was actually told about.
    uint64_t load_end_calls;

    // Times unified_cache_release_model_slot() returned, i.e. the number of
    // model teardowns that actually reached the cache.
    uint64_t release_slot_calls;

    // Slot argument of the most recent such call.  GGML_SYCL_MODEL_SLOT_NONE
    // until the corresponding call has happened once.  Lets a test check the
    // TOKEN identity -- that the slot a model was handed at load is the slot its
    // destructor released -- not merely that some call occurred.
    uint32_t load_end_last_slot;
    uint32_t release_slot_last_slot;

    // Entries the most recent release_model_slot() reclaimed.  Diagnostic only:
    // zero is a legitimate outcome (another live model may own every entry), so
    // do not gate on it.
    uint64_t release_slot_last_reclaimed;
};

// Read the counters.  NOT THREAD-SAFE: the fields are loaded one at a time, so a
// concurrent writer can land between two loads and yield a new call count beside
// a stale slot.  This is for single-threaded test use only -- do not build a
// concurrency check on it without giving it real ordering first.
GGML_BACKEND_API void ggml_backend_sycl_model_lifecycle_probe_read(
    struct ggml_backend_sycl_model_lifecycle_probe * out);

#ifdef __cplusplus
}
#endif
