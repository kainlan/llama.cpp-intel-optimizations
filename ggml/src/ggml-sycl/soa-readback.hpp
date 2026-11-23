//
// The inverse of the AOS->SOA weight reorder, for reading a weight back.
//
// set_tensor may store a quantized weight as SOA: every field of the block becomes one plane, and a plane holds
// that field of all blocks in block order. The planes follow each other without padding, so the SOA image of n
// blocks is exactly as large as their AOS image. ggml_backend_tensor_get returns ggml's block bytes, so a read
// of an SOA weight rebuilds them from the planes.
//
// ggml_sycl_soa_plane_table_for_type lists, for each type of soa-reorder-types.hpp, the planes in SOA order with
// the field's offset inside the AOS block. The fills in ggml-sycl.cpp (reorder_q4_0_cpu, reorder_q8_0_cpu,
// reorder_q4_k_cpu, reorder_q6_k_cpu, reorder_mxfp4_cpu and their device forms) write this layout.
// tests/test-sycl-soa-readback.cpp holds the numbers to the block structs and this table to that type list.
//
// Pure C++ over the ggml_type enum: usable from a host-only test.
//
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

struct ggml_sycl_soa_plane {
    size_t aos_offset;  // offset of the field inside one AOS block
    size_t width;       // bytes of the field per block
};

struct ggml_sycl_soa_plane_table {
    const ggml_sycl_soa_plane * planes;       // in SOA order
    size_t                      n_planes;
    size_t                      block_bytes;  // size of one AOS block = sum of the widths
};

inline bool ggml_sycl_soa_plane_table_for_type(ggml_type type, ggml_sycl_soa_plane_table * table) {
    // Each entry is { offset in the AOS block, width }. Plane order: Q4_0 and Q8_0 qs, d; Q4_K qs, scales, dm;
    // Q6_K ql, qh, scales, d; MXFP4 qs, e.
    static constexpr ggml_sycl_soa_plane q4_0[] = {
        { 2, 16 },
        { 0, 2  }
    };
    static constexpr ggml_sycl_soa_plane q8_0[] = {
        { 2, 32 },
        { 0, 2  }
    };
    static constexpr ggml_sycl_soa_plane q4_k[] = {
        { 16, 128 },
        { 4,  12  },
        { 0,  4   }
    };
    static constexpr ggml_sycl_soa_plane q6_k[] = {
        { 0,   128 },
        { 128, 64  },
        { 192, 16  },
        { 208, 2   }
    };
    static constexpr ggml_sycl_soa_plane mxfp4[] = {
        { 1, 16 },
        { 0, 1  }
    };
    switch (type) {
        case GGML_TYPE_Q4_0:
            *table = { q4_0, 2, 18 };
            return true;
        case GGML_TYPE_Q8_0:
            *table = { q8_0, 2, 34 };
            return true;
        case GGML_TYPE_Q4_K:
            *table = { q4_k, 3, 144 };
            return true;
        case GGML_TYPE_Q6_K:
            *table = { q6_k, 4, 210 };
            return true;
        case GGML_TYPE_MXFP4:
            *table = { mxfp4, 2, 17 };
            return true;
        default:
            return false;
    }
}

// Writes bytes [window_offset, window_offset + window_size) of the AOS image of n_blocks blocks to dst, reading
// the SOA image of the same n_blocks blocks at soa. Returns false, and writes nothing, for a type without a
// plane table or a window that leaves the image.
inline bool ggml_sycl_soa_to_aos_window(ggml_type    type,
                                        const void * soa,
                                        size_t       n_blocks,
                                        size_t       window_offset,
                                        size_t       window_size,
                                        void *       dst) {
    ggml_sycl_soa_plane_table table{};
    if (!ggml_sycl_soa_plane_table_for_type(type, &table)) {
        return false;
    }
    const size_t image_bytes = n_blocks * table.block_bytes;
    if (window_offset > image_bytes || window_size > image_bytes - window_offset) {
        return false;
    }
    if (window_size == 0) {
        return true;
    }
    const uint8_t * src         = static_cast<const uint8_t *>(soa);
    uint8_t *       out         = static_cast<uint8_t *>(dst);
    const size_t    window_end  = window_offset + window_size;
    const size_t    block_first = window_offset / table.block_bytes;
    const size_t    block_last  = (window_end - 1) / table.block_bytes;  // inclusive
    size_t          plane_base  = 0;
    for (size_t p = 0; p < table.n_planes; ++p) {
        const ggml_sycl_soa_plane & plane = table.planes[p];
        for (size_t b = block_first; b <= block_last; ++b) {
            // The field of block b is AOS bytes [field_begin, field_end); copy its part inside the window.
            const size_t field_begin = b * table.block_bytes + plane.aos_offset;
            const size_t field_end   = field_begin + plane.width;
            const size_t lo          = field_begin > window_offset ? field_begin : window_offset;
            const size_t hi          = field_end < window_end ? field_end : window_end;
            if (lo < hi) {
                std::memcpy(out + (lo - window_offset), src + plane_base + b * plane.width + (lo - field_begin),
                            hi - lo);
            }
        }
        plane_base += n_blocks * plane.width;
    }
    return true;
}
