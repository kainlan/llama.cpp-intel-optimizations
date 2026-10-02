#pragma once

// llama.cpp-kpjw: the scheduler's compute-allocation scope, host-testable on its own (tests/test-zone-sizing.cpp).
//
// A buffer the SYCL buffer type places while the scope is open on the calling thread is a scheduler compute buffer:
// the llama context opens it around every allocation its scheduler makes (reserve, graph alloc, the K-shift graph),
// and nothing else does. That is how a compute buffer is told from the model's weights and the recurrent state,
// which the same buffer type also backs -- by where the request comes from, never by when it is made.
//
// A per-thread depth, never negative: a leave with no matching enter (a guard unwound past its enter) leaves the
// scope closed rather than pushing the depth below zero, where the next enter would open nothing.

namespace ggml_sycl {

inline int & compute_alloc_scope_depth() {
    static thread_local int depth = 0;
    return depth;
}

inline void compute_alloc_scope_enter() {
    compute_alloc_scope_depth()++;
}

inline void compute_alloc_scope_leave() {
    if (compute_alloc_scope_depth() > 0) {
        compute_alloc_scope_depth()--;
    }
}

inline bool compute_alloc_scope_active() {
    return compute_alloc_scope_depth() > 0;
}

}  // namespace ggml_sycl
