#pragma once

#include <cstddef>

// Resolution of LLAMA_LAZY_MODE_AUTO (llama.cpp-z5fn).
//
// A lazily read tensor is ALWAYS placed in the CPU buffer type
// (llama_model_loader::lazy_read::buft()), so it is the CPU that executes and
// owns it. Upstream's rule -- AUTO falls back to OFF as soon as any device
// reports no mmap support (#28160, integrated GPUs) -- exists because such a
// device can consume host-pointer buffers and would be handed pages it cannot
// use. A backend whose placement planner decides residency itself (SYCL) does
// not do that: a planner-host tensor executes on the CPU backend, so that
// device's missing mmap_support says nothing about lazy tensors. Without this
// exemption SYCL never reports mmap_support, AUTO resolves to OFF, and a
// GET_ROWS-only table such as Qwen3.8's per_layer_token_embd (27.5 GB) is read
// into anonymous RAM instead of staying a file-backed, reclaimable mapping.
struct llama_lazy_device_caps {
    bool mmap_support;
    // The backend's own placement planner decides where weights live, and
    // executes planner-host weights on the CPU backend.
    bool planner_owns_placement;
};

static inline bool llama_lazy_auto_enabled(const llama_lazy_device_caps * devs, size_t n_devs) {
    for (size_t i = 0; i < n_devs; ++i) {
        if (!devs[i].mmap_support && !devs[i].planner_owns_placement) {
            return false;
        }
    }
    return true;
}
