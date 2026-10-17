//
// MIT license
// SPDX-License-Identifier: MIT
//
// Flash attention over a q4_0 / q8_0 KV cache on the Xe2 matrix engine (DPAS). The kernel is Torchit1's
// (https://github.com/Torchit1/llama.cpp, branch sycl-fa-q4-xmx, commit 24a3a71896cdf1d21b625c13c63f98b05bdde8e7);
// fattn-dpas.cpp carries the attribution and the list of what this fork changed.
//

#ifndef GGML_SYCL_FATTN_DPAS_HPP
#define GGML_SYCL_FATTN_DPAS_HPP

#include "fattn-common.hpp"

#include <cstddef>
#include <cstdint>

// Query heads per KV head that one work-group row tile holds.
#define GGML_SYCL_FATTN_DPAS_MAX_G 32

// Head sizes with a compiled kernel: multiples of the 32-dim q4_0 / q8_0 block.
inline bool ggml_sycl_fattn_dpas_head_ok(int64_t d) {
    return d == 64 || d == 96 || d == 128 || d == 192 || d == 256 || d == 512;
}

// What a launch chose, for the dispatch-debug line.
struct ggml_sycl_fattn_dpas_launch_info {
    int  nsplit = 1;      // key slices; > 1 means the split-K workspace was used
    int  nr16   = 0;      // 16-row tiles per work-group
    int  st     = 0;      // K/V staging: 0 per lane, 1 16-byte, 2 dword
    bool lg     = false;  // 256-register file
};

// Split-K workspace bound. A launch splits the keys only while all its work-groups fit one wave of
// xe_cores * (NR16 == 1 ? 4 : 2) work-groups, and a work-group covers at most 16 * NR16 (token, query head) rows, so a
// launch never holds more than 64 * xe_cores rows of split partials: 64 * xe_cores * D floats of output (D <= 512) and
// 64 * xe_cores (max, sum) pairs. Sized once per context from these, the workspace never has to grow.
inline size_t ggml_sycl_fattn_dpas_parts_capacity(int xe_cores) {
    return (size_t) 64 * (size_t) xe_cores * 512;
}

inline size_t ggml_sycl_fattn_dpas_meta_capacity(int xe_cores) {
    return (size_t) 64 * (size_t) xe_cores;
}

// Runs FLASH_ATTN_EXT with q4_0 or q8_0 K and V (one type for both) on stream. params is ggml_sycl_flash_attn_ext's
// filled fattn_params for an op ggml_sycl_fattn_dpas_supported() (fattn.cpp) admitted; device is the stream's device.
// parts / meta are the split-K workspace and its capacities in elements: when null or too small for the split a shape
// wants, the keys are not split. Returns the event of the last kernel submitted; the caller keeps the workspace alive
// until that event completes.
sycl::event ggml_sycl_fattn_dpas_launch(dpct::queue_ptr                    stream,
                                        int                                device,
                                        const fattn_params &               params,
                                        float *                            parts,
                                        sycl::float2 *                     meta,
                                        size_t                             parts_capacity,
                                        size_t                             meta_capacity,
                                        int                                xe_cores,
                                        ggml_sycl_fattn_dpas_launch_info * info);

#endif  // GGML_SYCL_FATTN_DPAS_HPP
