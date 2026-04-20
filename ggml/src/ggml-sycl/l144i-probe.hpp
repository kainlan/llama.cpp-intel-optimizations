//
// l144i-probe.hpp — non-determinism bisect instrumentation for 20B MoE garble
//
// Bead: llama.cpp-l144i
//
// Signature: gpt-oss-20b TG produces DIFFERENT tokens across runs with the same
// --seed 42 --temp 0.  Classic race-condition signature.  Predecessor agents
// ruled out:
//   - SYCL MXFP4 dense MUL_MAT math (34/34 tests pass)
//   - ggml-cpu MXFP4 MUL_MAT_ID (618/618 tests pass incl. 20B shape)
//   - Fusion path (disabled under placement plan)
//   - CPU-TG fast-path (non-determinism persists with GGML_SYCL_CPU_EXPERT_TG=0)
//   - Graph replay (persists with GGML_SYCL_DISABLE_GRAPH=1)
//   - SAFE_MODE drain-after-op (persists with GGML_SYCL_SAFE_MODE=1)
//   - Legacy unified dispatch (persists with GGML_SYCL_UNIFIED_FORCE_LEGACY=1)
//
// The race must be in host-side state visible to subsequent ops.  This probe
// captures MUL_MAT_ID output tensor contents into a stable per-call FNV hash so
// we can correlate which ops' outputs differ across runs.
//
// Compile-gate: GGML_SYCL_L144I_PROBE.  Runtime-gate: GGML_SYCL_L144I_PROBE=1.
// Default: compiled-in (overhead ~1 microsecond/op) but OFF at runtime.
//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_L144I_PROBE_HPP
#define GGML_SYCL_L144I_PROBE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef GGML_SYCL_L144I_PROBE
#    define GGML_SYCL_L144I_PROBE 1
#endif

#if GGML_SYCL_L144I_PROBE

namespace ggml_sycl {
namespace l144i {

// Runtime gate, cached at first call.  GGML_SYCL_L144I_PROBE=1 to enable.
inline bool enabled() {
    static std::atomic<int> state{ -1 };
    int val = state.load(std::memory_order_acquire);
    if (val < 0) {
        const char * env = std::getenv("GGML_SYCL_L144I_PROBE");
        int          new_val = (env && std::atoi(env) != 0) ? 1 : 0;
        state.compare_exchange_strong(val, new_val, std::memory_order_release, std::memory_order_acquire);
        val = state.load(std::memory_order_acquire);
    }
    return val != 0;
}

// Maximum tokens to probe (0 means unlimited).  GGML_SYCL_L144I_PROBE_MAX_TOK=N.
inline int max_tokens() {
    static std::atomic<int> state{ -1 };
    int val = state.load(std::memory_order_acquire);
    if (val < 0) {
        const char * env = std::getenv("GGML_SYCL_L144I_PROBE_MAX_TOK");
        int          new_val = env ? std::atoi(env) : 4;
        state.compare_exchange_strong(val, new_val, std::memory_order_release, std::memory_order_acquire);
        val = state.load(std::memory_order_acquire);
    }
    return val;
}

// FNV-1a 64-bit hash of the raw bytes.  Deterministic — same bytes → same hash.
inline uint64_t fnv_hash(const void * data, std::size_t bytes) {
    const uint8_t * p    = static_cast<const uint8_t *>(data);
    uint64_t        hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Finite-element-count hash: stride over the buffer, hashing a sample of
// bytes, to keep per-call overhead bounded regardless of tensor size.  Still
// fully deterministic.  Returns (hash, n_nonfinite).
struct sample_result {
    uint64_t hash;
    int      n_nan;
    int      n_inf;
    float    min_abs;
    float    max_abs;
};

inline sample_result sample_floats(const float * p, std::size_t n_floats) {
    sample_result r{ 14695981039346656037ULL, 0, 0, 1e30f, 0.0f };
    if (!p || n_floats == 0) {
        return r;
    }
    // Hash every element (keeps sensitivity to tiny changes); the caller
    // decides how often to probe.
    for (std::size_t i = 0; i < n_floats; i++) {
        float    v = p[i];
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        r.hash ^= u;
        r.hash *= 1099511628211ULL;
        if (v != v) {
            r.n_nan++;
            continue;
        }
        uint32_t ubits = u & 0x7fffffffU;
        if (ubits == 0x7f800000U) {
            r.n_inf++;
            continue;
        }
        float a = v < 0 ? -v : v;
        if (a < r.min_abs) r.min_abs = a;
        if (a > r.max_abs) r.max_abs = a;
    }
    return r;
}

// Global counters keyed by (tensor_name, call_site).  Advance per call so we
// can correlate across runs: call #0 of ffn_moe_down-16 should have the same
// hash on two runs if compute is deterministic.
//
// Implementation is minimal: a header-only FILE* log to stderr, with a process-
// local monotonic call index.  For bead purposes we do not need to compare at
// runtime — the user diffs the stderr logs from two runs.
inline int next_call_index() {
    static std::atomic<int> counter{ 0 };
    return counter.fetch_add(1, std::memory_order_relaxed);
}

// Emit a probe line.  The calling site must hold information about tensor name,
// site tag, and probe-extent.  Per-call index is auto-advanced.
inline void log_output_sample(const char *   site,
                              const char *   tensor_name,
                              int            token_index,
                              int            layer_id,
                              const float *  data,
                              std::size_t    n_floats) {
    if (!enabled()) {
        return;
    }
    int max_tok = max_tokens();
    if (max_tok > 0 && token_index >= max_tok) {
        return;
    }
    int idx = next_call_index();
    sample_result s = sample_floats(data, n_floats);
    std::fprintf(stderr,
                 "[L144I] %s tok=%d layer=%d tensor=%s idx=%d n=%zu hash=0x%016lx "
                 "nan=%d inf=%d min_abs=%.4g max_abs=%.4g\n",
                 site,
                 token_index,
                 layer_id,
                 tensor_name ? tensor_name : "(null)",
                 idx,
                 n_floats,
                 (unsigned long) s.hash,
                 s.n_nan,
                 s.n_inf,
                 (double) s.min_abs,
                 (double) s.max_abs);
    std::fflush(stderr);
}

// Simpler variant: hash arbitrary bytes without float breakdown.
inline void log_bytes_sample(const char *  site,
                             const char *  tensor_name,
                             int           token_index,
                             int           layer_id,
                             const void *  data,
                             std::size_t   bytes) {
    if (!enabled()) {
        return;
    }
    int max_tok = max_tokens();
    if (max_tok > 0 && token_index >= max_tok) {
        return;
    }
    int idx = next_call_index();
    uint64_t h = fnv_hash(data, bytes);
    std::fprintf(stderr,
                 "[L144I] %s tok=%d layer=%d tensor=%s idx=%d bytes=%zu hash=0x%016lx\n",
                 site,
                 token_index,
                 layer_id,
                 tensor_name ? tensor_name : "(null)",
                 idx,
                 bytes,
                 (unsigned long) h);
    std::fflush(stderr);
}

// Log a raw scalar value.  Useful for checking if a "should be stable" value
// (e.g. expert_id routing decision) varies across runs.
inline void log_scalar(const char * site,
                       const char * tensor_name,
                       int          token_index,
                       int          layer_id,
                       const char * label,
                       long long    value) {
    if (!enabled()) {
        return;
    }
    int max_tok = max_tokens();
    if (max_tok > 0 && token_index >= max_tok) {
        return;
    }
    std::fprintf(stderr,
                 "[L144I] %s tok=%d layer=%d tensor=%s %s=%lld\n",
                 site,
                 token_index,
                 layer_id,
                 tensor_name ? tensor_name : "(null)",
                 label ? label : "value",
                 value);
    std::fflush(stderr);
}

}  // namespace l144i
}  // namespace ggml_sycl

#    define GGML_SYCL_L144I_PROBE_FLOATS(site, tname, tok, layer, ptr, n)                   \
        ::ggml_sycl::l144i::log_output_sample((site), (tname), (tok), (layer), (ptr), (n))
#    define GGML_SYCL_L144I_PROBE_BYTES(site, tname, tok, layer, ptr, n)                    \
        ::ggml_sycl::l144i::log_bytes_sample((site), (tname), (tok), (layer), (ptr), (n))
#    define GGML_SYCL_L144I_PROBE_SCALAR(site, tname, tok, layer, label, value)             \
        ::ggml_sycl::l144i::log_scalar((site), (tname), (tok), (layer), (label), (value))

#else  // GGML_SYCL_L144I_PROBE == 0

#    define GGML_SYCL_L144I_PROBE_FLOATS(site, tname, tok, layer, ptr, n) ((void) 0)
#    define GGML_SYCL_L144I_PROBE_BYTES(site, tname, tok, layer, ptr, n)  ((void) 0)
#    define GGML_SYCL_L144I_PROBE_SCALAR(site, tname, tok, layer, label, value) ((void) 0)

#endif  // GGML_SYCL_L144I_PROBE

#endif  // GGML_SYCL_L144I_PROBE_HPP
