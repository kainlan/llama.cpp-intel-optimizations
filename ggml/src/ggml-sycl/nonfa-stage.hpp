#pragma once

// The non-FA attention path's batched f16 mul_mat: which MUL_MAT nodes reach
// ggml_sycl_mul_mat_batched_sycl, and how many f16 elements it stages for src1.
//
// ggml_sycl_mul_mat walks the route chain and the batched op sizes its staging
// buffer; the measure pass's context-nonfa-stage visitor sizes the slot a
// planned context carves for that staging. All three call the functions below,
// so the slot and the site cannot count one shape two ways (zhcn design 2.8: the
// measure sizes a demand with the runtime site's own function, never a second derivation).
//
// This header names only ggml types, so a host test builds it without a device.
// The facts that are not a function of the tensors -- whether src0 is in a
// row-split buffer, whether an operand is a weight, the debug overrides and
// which staging path is compiled -- arrive in `ggml_sycl_mul_mat_route_env`,
// which ggml-sycl.cpp fills in one function for the dispatch and for the measure
// view alike, through ggml_sycl_mul_mat_route_env_from below.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

// sizeof(sycl::half), named here because this header has no sycl. ggml-sycl.cpp
// static_asserts it against the real type.
#define GGML_SYCL_NONFA_STAGE_ELEM_BYTES 2

struct ggml_sycl_mul_mat_route_env {
    bool split            = false;  // src0 is in a row-split buffer
    bool has_weight       = false;  // src0 or src1 is a weight (buffer usage WEIGHTS)
    bool kqv_force_simple = false;  // GGML_SYCL_KQV_FORCE_SIMPLE or GGML_SYCL_KQV_DISABLE_FP16
    bool stage_strided    = false;  // the oneDNN-strided staging path, not the oneMath element-count one
};

// Whether src0 is in a row-split buffer. A tensor with no buffer is in none, and `is_split_buffer`, the backend's
// own predicate (it reads the buffer's type), is never asked about it. Every read of this fact in the mul_mat
// dispatch goes through here.
inline bool ggml_sycl_mul_mat_src0_is_split(const ggml_tensor * src0, bool (*is_split_buffer)(ggml_backend_buffer_t)) {
    return src0->buffer != nullptr && is_split_buffer(src0->buffer);
}

// The environment of `mul_mat(src0, src1)`. A tensor with no buffer is in no row-split buffer and is no weight:
// the buffer predicates are never asked about it. That is the answer for an operand the scheduler has not placed
// yet, which includes a weight at a load-time MEASURE; the route then reads such a weight as an activation, so a
// walker that measures before weights have buffers must set `has_weight` itself (see context_measure_view).
// `is_split_buffer` and `is_weight_tensor` are the backend's own predicates, passed so this header needs no device.
inline ggml_sycl_mul_mat_route_env ggml_sycl_mul_mat_route_env_from(const ggml_tensor * src0,
                                                                    const ggml_tensor * src1,
                                                                    bool (*is_split_buffer)(ggml_backend_buffer_t),
                                                                    bool (*is_weight_tensor)(const ggml_tensor *),
                                                                    bool kqv_force_simple,
                                                                    bool stage_strided) {
    ggml_sycl_mul_mat_route_env env;
    env.split = ggml_sycl_mul_mat_src0_is_split(src0, is_split_buffer);
    env.has_weight =
        (src0->buffer != nullptr && is_weight_tensor(src0)) || (src1->buffer != nullptr && is_weight_tensor(src1));
    env.kqv_force_simple = kqv_force_simple;
    env.stage_strided    = stage_strided;
    return env;
}

// The branch of the f16 attention chain a MUL_MAT node takes. The values are the
// chain's own branches, in the order it tests them.
enum ggml_sycl_mul_mat_f16_route {
    GGML_SYCL_MUL_MAT_F16_ROUTE_NONE,          // none of the three: the orchestrator's dispatch
    GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_P021,       // single-token single-sequence KQ: the p021 kernel
    GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_BATCHED,    // the same branch, any other batch: batched
    GGML_SYCL_MUL_MAT_F16_ROUTE_VEC_NC,        // single-token KQV over a strided view: the nc kernel
    GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_BATCHED,  // multi-batch KQ and KQV: batched
    GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_SCALAR,   // the same branch under the KQV debug override: scalar fallback
};

// A kqv matmul by name, which the KQV debug overrides key on.
inline bool ggml_sycl_mul_mat_is_kqv(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (dst && dst->name[0] != '\0' && std::strstr(dst->name, "kqv") != nullptr) {
        return true;
    }
    if (src0 && src0->name[0] != '\0' && std::strstr(src0->name, "cache_v") != nullptr) {
        return true;
    }
    if (src1 && src1->name[0] != '\0' && std::strstr(src1->name, "kq_soft_max") != nullptr) {
        return true;
    }
    return false;
}

// The branch `dst = mul_mat(src0, src1)` takes. ggml_sycl_mul_mat switches on
// this; nothing else decides it.
inline ggml_sycl_mul_mat_f16_route ggml_sycl_mul_mat_f16_route_of(const ggml_tensor *                 src0,
                                                                  const ggml_tensor *                 src1,
                                                                  const ggml_tensor *                 dst,
                                                                  const ggml_sycl_mul_mat_route_env & env) {
    if (env.split || src0->type != GGML_TYPE_F16) {
        return GGML_SYCL_MUL_MAT_F16_ROUTE_NONE;
    }
    if (!env.has_weight && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // The p021 kernel is specific to the single-batch dimensions.
        return src0->ne[3] == 1 && src1->ne[3] == 1 ? GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_P021 :
                                                      GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_BATCHED;
    }
    if (!ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        return GGML_SYCL_MUL_MAT_F16_ROUTE_VEC_NC;
    }
    if (!env.has_weight && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        return env.kqv_force_simple && ggml_sycl_mul_mat_is_kqv(src0, src1, dst) ?
                   GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_SCALAR :
                   GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_BATCHED;
    }
    return GGML_SYCL_MUL_MAT_F16_ROUTE_NONE;
}

// Whether the node reaches ggml_sycl_mul_mat_batched_sycl.
inline bool ggml_sycl_mul_mat_routes_batched_f16(const ggml_tensor *                 src0,
                                                 const ggml_tensor *                 src1,
                                                 const ggml_tensor *                 dst,
                                                 const ggml_sycl_mul_mat_route_env & env) {
    const ggml_sycl_mul_mat_f16_route route = ggml_sycl_mul_mat_f16_route_of(src0, src1, dst, env);
    return route == GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_BATCHED || route == GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_BATCHED;
}

// The f16 element count ggml_sycl_mul_mat_batched_sycl's staging alloc() receives
// for src1, and 0 when src1 is already f16 and nothing is staged. `strided` is
// the oneDNN path, which converts the strided span; the oneMath path converts
// the elements.
inline size_t ggml_sycl_batched_f16_src1_stage_elems(const ggml_tensor * src1, bool strided) {
    if (src1->type == GGML_TYPE_F16) {
        return 0;
    }
    if (!strided) {
        return (size_t) ggml_nelements(src1);
    }
    // Iterate the dims and find the slowest moving dim and stride; the last
    // stride is always the largest.
    int    last_dim    = 0;
    int    last_str    = 0;
    size_t largest_str = 0;
    for (int i = 0; i < 4; i++) {
        if (src1->nb[i] == largest_str) {
            if (src1->ne[last_dim] == 1) {
                last_str = i;
                last_dim = i;
            }
        }
        if (src1->nb[i] > largest_str) {
            largest_str = src1->nb[i];
            last_str    = i;
            last_dim    = i;
        }
    }
    return (size_t) (src1->nb[last_str] * src1->ne[last_dim] / ggml_type_size(src1->type));
}
