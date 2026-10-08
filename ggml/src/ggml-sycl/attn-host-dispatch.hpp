//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

// Deliberately no #include of unified-cache.hpp or any SYCL header (the
// c-8lff siting lesson, TKV-4/kv-runtime-demotion precedent): this is a pure,
// host-linkable TU so it can be unit-tested without oneAPI. It depends only
// on ggml.h (ggml_tensor / GGML_MAX_SRC), which itself pulls in no SYCL
// headers -- see ggml/src/ggml-sycl/tests/test-dmmv-coalesced-q4-0-oracle.cpp
// for the established "links ggml-base only, no -fsycl" pattern this test
// target follows.
#include "ggml.h"

#include <cstddef>

namespace ggml_sycl {

// TKV-13 (B2) step 1: the DAG-consumption check that lets the SYCL backend
// defer -- rather than block on -- a pending host-computed result, the
// mechanism CpuExpertPool's MoE CPU dispatch uses to overlap with GPU work.
// ggml-sycl.cpp's ggml_sycl_op_consumes_tensor (the MoE expert-deferral flush
// check) calls attn_op_consumes_tensor here, so the walk is a pure TU that is
// unit-tested without oneAPI (docs/plans/2026-08-27-tkv13-b2-addendum.md §2,
// §6 step 1).

// Does `tensor` transitively read `target`, walking the view_src chain at
// every node and recursing into every ggml_tensor::src[]? Depth-bounded
// (`depth > 32`) and visited-set deduplicated, so a malformed graph with a
// src[] cycle terminates and converging paths cost their node count.
bool attn_tensor_depends_on(const ggml_tensor * tensor, const ggml_tensor * target, int depth = 0);

// Same answer as attn_tensor_depends_on; additionally reports in *visits (when
// non-null) how many distinct nodes the walk expanded. Each node is expanded at
// most once per call, so on a DAG with many converging paths (the residual
// stream) the cost is the reachable subgraph, not the number of paths. This is
// the test seam that pins that bound.
bool attn_tensor_depends_on_counted(const ggml_tensor * tensor, const ggml_tensor * target, int depth, size_t * visits);

// Is `pending_dst` one of `consuming_dst`'s actual inputs -- i.e. would
// dispatching `consuming_dst` today read data that a still-in-flight
// host-computed attention result would produce? Checked over
// consuming_dst->src[] only, deliberately never over consuming_dst itself:
// a producer that WRITES pending_dst (e.g. a fused op reusing the same
// tensor as its own output) is not a consumer of it and must not trigger a
// flush -- mirrors the existing "fused MoE producer" comment on
// ggml_sycl_op_consumes_tensor. A node for which this returns false must be
// dispatchable without waiting on the pending result at all (the whole point
// of the overlap); a node for which it returns true is the one genuine place
// a blocking flush belongs.
bool attn_op_consumes_tensor(const ggml_tensor * consuming_dst, const ggml_tensor * pending_dst);

}  // namespace ggml_sycl
