//
// Why the layer-block executor did or did not run a graph (llama.cpp-m1or).
//
// The block classifier used to print a literal `executor=inactive` whatever
// happened next, so the one log line meant to say whether the executor ran
// could not change. The executor now reports the first gate that stopped it,
// and the pre-execution gates are evaluated here -- by the same function the
// executor uses to decide -- so the reported gate and the real decision cannot
// drift apart.
//
// Dependency-free so it can be unit-tested on a host with no GPU
// (tests/test-sycl-split-exec-policy.cpp).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>

namespace ggml_sycl {

// Listed in evaluation order. NONE means the executor ran the graph.
enum block_exec_gate {
    BLOCK_EXEC_GATE_NONE = 0,
    BLOCK_EXEC_GATE_EXECUTE_DISABLED,      // GGML_SYCL_BLOCK_EXEC_EXECUTE unset or 0
    BLOCK_EXEC_GATE_NO_GRAPH,              // no cgraph to execute
    BLOCK_EXEC_GATE_GRAPH_RECORDING,       // a SYCL command graph is being recorded
    BLOCK_EXEC_GATE_NO_PLAN,               // no placement plan visible to the device
    BLOCK_EXEC_GATE_CANDIDATE_BLOCKS,      // fewer than 2 candidate layer blocks
    BLOCK_EXEC_GATE_ACTIVE_MULTI_BLOCK,    // the active plan already spans several blocks
    BLOCK_EXEC_GATE_PREPARE_CONTEXT,       // per-graph execution context could not be prepared
    BLOCK_EXEC_GATE_SMALL_BOUNDARY,        // largest boundary below GGML_SYCL_BLOCK_EXEC_MIN_BOUNDARY_BYTES
    BLOCK_EXEC_GATE_SECONDARY_PROMPT_MOE,  // prompt MoE on a secondary device is unsupported
    BLOCK_EXEC_GATE_EXECUTION_REJECTED,    // a node failed after execution began
};

struct block_exec_precheck_inputs {
    bool   execute_enabled  = false;
    bool   has_graph        = false;
    bool   graph_recording  = false;
    bool   has_plan         = false;
    size_t candidate_blocks = 0;
    size_t active_blocks    = 0;
};

// The first gate, among those decidable before any per-graph preparation,
// that stops the executor; BLOCK_EXEC_GATE_NONE when all of them pass.
inline block_exec_gate block_exec_first_failing_precheck(const block_exec_precheck_inputs & in) {
    if (!in.execute_enabled) {
        return BLOCK_EXEC_GATE_EXECUTE_DISABLED;
    }
    if (!in.has_graph) {
        return BLOCK_EXEC_GATE_NO_GRAPH;
    }
    if (in.graph_recording) {
        return BLOCK_EXEC_GATE_GRAPH_RECORDING;
    }
    if (!in.has_plan) {
        return BLOCK_EXEC_GATE_NO_PLAN;
    }
    if (in.candidate_blocks < 2) {
        return BLOCK_EXEC_GATE_CANDIDATE_BLOCKS;
    }
    if (in.active_blocks > 1) {
        return BLOCK_EXEC_GATE_ACTIVE_MULTI_BLOCK;
    }
    return BLOCK_EXEC_GATE_NONE;
}

inline const char * block_exec_gate_name(block_exec_gate gate) {
    switch (gate) {
        case BLOCK_EXEC_GATE_NONE:
            return "none";
        case BLOCK_EXEC_GATE_EXECUTE_DISABLED:
            return "execute-disabled";
        case BLOCK_EXEC_GATE_NO_GRAPH:
            return "no-graph";
        case BLOCK_EXEC_GATE_GRAPH_RECORDING:
            return "graph-recording";
        case BLOCK_EXEC_GATE_NO_PLAN:
            return "no-plan";
        case BLOCK_EXEC_GATE_CANDIDATE_BLOCKS:
            return "candidate-blocks";
        case BLOCK_EXEC_GATE_ACTIVE_MULTI_BLOCK:
            return "active-multi-block";
        case BLOCK_EXEC_GATE_PREPARE_CONTEXT:
            return "prepare-context";
        case BLOCK_EXEC_GATE_SMALL_BOUNDARY:
            return "small-boundary";
        case BLOCK_EXEC_GATE_SECONDARY_PROMPT_MOE:
            return "secondary-prompt-moe";
        case BLOCK_EXEC_GATE_EXECUTION_REJECTED:
            return "execution-rejected";
    }
    return "unknown";
}

}  // namespace ggml_sycl
