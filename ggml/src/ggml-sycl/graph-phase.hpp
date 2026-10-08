#pragma once

// Decode/prompt classification of one backend-scheduler split.
//
// The scheduler hands the backend one split at a time, and a MoE model can
// produce dozens of splits per token. Several of them carry no dense MUL_MAT:
// some hold only MUL_MAT_ID nodes, some only norms, adds or copies. A
// classifier that reads only dense MUL_MATs and calls everything else a prompt
// flips the phase several times per decode token, and every flip clears the
// active command graph and re-runs the prompt-to-decode transition work.
//
// So the evidence is the first matmul in node order, dense or routed:
//   - MUL_MAT: src1->ne[1] is the row count; one row is decode.
//   - MUL_MAT_ID: src[2] (ids) is [n_expert_used, n_tokens]; ids->ne[1] is the
//     routed-token count; one token is decode.
// A split with neither has no batch evidence at all, and the phase it runs in
// is the one the previous split established, so it keeps that phase.
//
// SYCL-free on purpose so tests/test-sycl-graph-phase.cpp can run it on
// synthetic ggml graphs without a device.

#include "ggml.h"

namespace ggml_sycl {

inline bool graph_phase_is_decode(ggml_tensor * const * nodes, int n_nodes, bool previous_is_decode) {
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * node = nodes[i];
        if (node->op == GGML_OP_MUL_MAT && node->src[1]) {
            return node->src[1]->ne[1] == 1;
        }
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[2]) {
            return node->src[2]->ne[1] == 1;
        }
    }
    return previous_is_decode;
}

}  // namespace ggml_sycl
