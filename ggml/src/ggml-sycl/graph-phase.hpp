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
// So the evidence, strongest first:
//   1. The first matmul in node order, dense or routed:
//      - MUL_MAT: src1->ne[1] is the row count; one row is decode.
//      - MUL_MAT_ID: src[2] (ids) is [n_expert_used, n_tokens]; ids->ne[1] is
//        the routed-token count; one token is decode.
//   2. Without a matmul, the first per-token lookup: a GET_ROWS from a leaf
//      2-D table (a token or position embedding) by a 1-D index vector. It
//      gathers one row per token, so one row is decode. This is what lets the
//      first, matmul-free split of a new prompt after decode tokens read as a
//      prompt. Gathers from a view are not lookups and are ignored: a
//      recurrent-state gather reshapes the state cache and indexes per
//      sequence, and the MoE weight gather reshapes the router output to 3-D
//      and indexes with 2-D ids. A gather of activations (out_ids) has a
//      computed, not leaf, table, except when the scheduler copied that
//      activation into this split as an input; the out_ids gather sits just
//      before its layer's FFN matmuls, which outrank it whenever they share
//      the split.
//   3. With neither, the split has no batch evidence at all, and the phase it
//      runs in is the one the previous split established, so it keeps it.
// Matmuls rank first so a split that has one is classified exactly as before.
//
// SYCL-free on purpose so tests/test-sycl-graph-phase.cpp can run it on
// synthetic ggml graphs without a device.

#include "ggml.h"

namespace ggml_sycl {

inline bool graph_phase_is_token_lookup(const ggml_tensor * node) {
    const ggml_tensor * table = node->src[0];
    const ggml_tensor * rows  = node->src[1];
    return node->op == GGML_OP_GET_ROWS && table && rows && table->op == GGML_OP_NONE && !table->view_src &&
           table->ne[2] == 1 && table->ne[3] == 1 && rows->ne[1] == 1 && rows->ne[2] == 1 && rows->ne[3] == 1;
}

inline bool graph_phase_is_decode(ggml_tensor * const * nodes, int n_nodes, bool previous_is_decode) {
    const ggml_tensor * lookup = nullptr;
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * node = nodes[i];
        if (node->op == GGML_OP_MUL_MAT && node->src[1]) {
            return node->src[1]->ne[1] == 1;
        }
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[2]) {
            return node->src[2]->ne[1] == 1;
        }
        if (!lookup && graph_phase_is_token_lookup(node)) {
            lookup = node;
        }
    }
    if (lookup) {
        return lookup->src[1]->ne[0] == 1;
    }
    return previous_is_decode;
}

}  // namespace ggml_sycl
