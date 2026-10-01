#pragma once

// The pure chunk-cap expression (zhcn-design §2.4).  A cap is the largest single
// buffer a device buft's get_max_size reports to ggml-alloc.  Every cap on the
// plan path is this one expression, so MEASURE and ALLOC, the freeze and the
// load-time scope agree by construction.  Host-only: it names no device, backing
// or cache, so a test can call it with seamed capacities.

#include <algorithm>
#include <cstddef>

constexpr size_t GGML_SYCL_CHUNK_CAP_MAX = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // the 2 GiB safe L0 cap

struct ggml_sycl_chunk_cap_result {
    size_t       cap     = 0;
    // Non-null when no allocation limit is known: both A inputs are 0.  The cap is then
    // not bounded by A, and the caller names the refusal with its device.
    const char * refusal = nullptr;
};

// USM:  min(2 GiB, A).
// VM:   min(max(runtime, kv, scratch), 2 GiB, A); all three capacities 0 is read as
//       "no capacity configured yet" and answers the 2 GiB bound, as the unscoped
//       read always has.
// A is safe_alloc, or max_alloc when safe_alloc is 0.  Both 0: the named refusal.
// A USM caller passes 0 for the three capacities and nothing reads them.
inline ggml_sycl_chunk_cap_result ggml_sycl_chunk_cap_core(bool   is_vm,
                                                           size_t runtime,
                                                           size_t kv,
                                                           size_t scratch,
                                                           size_t safe_alloc,
                                                           size_t max_alloc) {
    ggml_sycl_chunk_cap_result result;
    result.cap = GGML_SYCL_CHUNK_CAP_MAX;
    if (is_vm) {
        const size_t planned = std::max({ runtime, kv, scratch });
        if (planned > 0) {
            result.cap = std::min(result.cap, planned);
        }
    }
    const size_t allocation_limit = safe_alloc > 0 ? safe_alloc : max_alloc;
    if (allocation_limit > 0) {
        result.cap = std::min(result.cap, allocation_limit);
    } else {
        result.refusal = "no allocation limit known";
    }
    return result;
}
