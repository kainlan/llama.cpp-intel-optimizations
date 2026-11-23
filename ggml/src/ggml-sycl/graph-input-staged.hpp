//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_GRAPH_INPUT_STAGED_HPP
#define GGML_SYCL_GRAPH_INPUT_STAGED_HPP

// How a kernel being recorded reads a graph input.
//
// llama writes a graph input on the host every ubatch, and a recorded kernel bakes the pointers it is given, so it
// must not be given the input's host bytes. The graph pre-stage copies every input a computed node reads to a stable
// device buffer before the token runs (graph_prestage_leaf_tensors, INPUT arm, keyed by tensor identity in
// graph_input_staging) and refreshes that copy before every replay. These are the one lookup of that copy every op
// uses, so no op's idea of which tensors are staged, or for how many bytes, can drift from the pre-stage's or from
// another op's.

#include "common.hpp"
#include "prestage-class.hpp"

#include <stdexcept>
#include <string>

// The device copy of a graph input that the graph pre-stage made this token, or nullptr. A kernel being recorded reads
// that copy instead of the input's host bytes: raw host staging cannot be recorded, and running the segment directly
// instead pays a stream drain for the staging copy.
//
// The handle comes back in out_handle so the caller can keep it alive for as long as the recorded kernel can read it.
static inline void * ggml_sycl_graph_input_staged_ptr(ggml_backend_sycl_context & ctx,
                                                      const ggml_tensor *         tensor,
                                                      size_t                      span_bytes,
                                                      int                         device,
                                                      ggml_sycl::mem_handle *     out_handle) {
    // Only while this thread records. The staged copy is current only when the pre-stage ran for this token; a direct
    // run has no such guarantee, and its raw host staging is correct there.
    if (!ggml_sycl_graph_recording_this_thread() || tensor == nullptr || tensor->name[0] == '\0' || span_bytes == 0) {
        return nullptr;
    }
    // Only a flagged graph input or a view of one, by the pre-stage's own INPUT rule, so this never accepts a tensor
    // the pre-stage does not stage and refresh. A host op output from another split is staged as an input too, but
    // the caller cannot tell it from an output this graph produces, whose staging entry would predate its producer, so
    // that case keeps failing closed.
    if (!ggml_sycl_prestage_is_input(tensor)) {
        return nullptr;
    }
    // The pre-stage stages a source under the tensor pointer it meets, copying ggml_nbytes(tensor) from that tensor's
    // own data, so the copy starts where the kernel's addressing starts. It stages a storage range once per pass,
    // keyed by (cache id, device, view offset), so a second tensor over the same range gets no entry of its own: the
    // lookup then misses and the caller fails closed, which costs speed, not correctness. The entry's capacity may
    // exceed what was copied this token (a reused, larger buffer), so the span must also fit the copied bytes.
    if (span_bytes > ggml_nbytes(tensor)) {
        return nullptr;
    }
    void * staged_ptr = nullptr;
    if (!ctx.graph_input_stage_lookup(tensor, span_bytes, device, out_handle, &staged_ptr)) {
        return nullptr;
    }
    return staged_ptr;
}

// The pointer a kernel with no raw host staging of its own may bake for a source it reads, or nullptr for an absent
// source.
//
// While this thread records: the pre-stage's copy of a graph input (its handle in out_handle, for the caller's
// terminal owners), else the tensor's own memory when that is device USM. Anything else throws std::runtime_error, and
// the caller resolves every source before it submits anything, so the recorder drops the segment to direct dispatch
// with nothing half-recorded. That covers a miss on a host input (a second tensor over a range the pre-stage staged
// once gets no entry of its own): its host-pinned pointer would have the GPU read host memory, which placement
// forbids, and a recording would keep reading it after llama re-points the input's data; a pointer outside USM the
// GPU cannot read at all.
//
// Outside a recording: the resolved pointer, as every op resolves it.
static inline void * ggml_sycl_graph_input_recordable_ptr(ggml_backend_sycl_context & ctx,
                                                          const ggml_tensor *         tensor,
                                                          size_t                      span_bytes,
                                                          int                         device,
                                                          ggml_sycl::mem_handle &     out_handle,
                                                          const char *                op_name) {
    if (tensor == nullptr) {
        return nullptr;
    }
    if (!ggml_sycl_graph_recording_this_thread()) {
        return ggml_sycl_resolve_tensor_ptr(tensor, device);
    }
    ggml_sycl::mem_handle staged_handle;
    if (void * staged = ggml_sycl_graph_input_staged_ptr(ctx, tensor, span_bytes, device, &staged_handle)) {
        out_handle = std::move(staged_handle);
        return staged;
    }
    void * ptr = ggml_sycl_resolve_tensor_ptr(tensor, device);
    if (ptr == nullptr || ggml_sycl_get_alloc_type(ptr) != sycl::usm::alloc::device) {
        throw std::runtime_error(std::string(op_name) + ": source " + tensor->name +
                                 " is not device memory and has no staged copy, so it is not graph-recordable");
    }
    return ptr;
}

#endif  // GGML_SYCL_GRAPH_INPUT_STAGED_HPP
