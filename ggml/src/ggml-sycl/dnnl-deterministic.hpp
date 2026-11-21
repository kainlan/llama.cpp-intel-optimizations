#pragma once

// This header depends on oneDNN and the C library only, so that the standalone oneDNN benchmark, which does not link
// the backend, builds its attributes through the same helper as the backend.

#include "dnnl.hpp"

#include <cstdlib>
#include <cstring>

// Without the deterministic attribute oneDNN may pick an implementation whose order of summation depends on how its
// work is scheduled, so identical inputs give different bits from one run to the next and greedy decoding parts where
// two candidates nearly tie. GGML_SYCL_ONEDNN_DETERMINISTIC=0 drops the attribute, so its cost can be measured by
// flipping a variable on one binary. Read once. The name is spelled once, here, for the accessor and its
// sycl_env_settings[] row in ggml_check_sycl().
static constexpr const char * GGML_SYCL_ONEDNN_DETERMINISTIC_ENV = "GGML_SYCL_ONEDNN_DETERMINISTIC";

// Only the exact value "0" turns the attribute off. An empty value, "false", "off" or a typo keeps it on: the
// reproducible mode is the safe side of a mistyped variable, and std::atoi would read every one of those as 0.
inline bool ggml_sycl_onednn_deterministic_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv(GGML_SYCL_ONEDNN_DETERMINISTIC_ENV);
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

// Every dnnl::primitive_attr the backend constructs is handed to this in the statement right after its construction,
// before any primitive descriptor is built from it. The weight prepack and its size query build descriptors too: they
// must carry the same attribute as the primitive that executes, or the packed layout could differ from the one it
// expects.
inline void ggml_sycl_dnnl_attr_set_reproducible(dnnl::primitive_attr & attr) {
    if (ggml_sycl_onednn_deterministic_enabled()) {
        attr.set_deterministic(true);
    }
}
