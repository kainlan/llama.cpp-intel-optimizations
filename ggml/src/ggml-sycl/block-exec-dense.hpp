//
// Dense-split layer-block executor: which device runs each node, and what
// must cross between devices (llama.cpp-tf8m).
//
// A dense model split over two cards (Mistral 7B: B70 layers 0-29, B50 layers
// 30-31) used to run every op of the second card through a per-op route: stage
// the operands, run, drain the target queue, publish. The executor instead cuts
// the graph into contiguous node ranges, one per device, and runs each range
// through the ordinary node loop on its own device. Only the tensors that cross
// a range edge are copied, once per graph.
//
// This header holds the decisions, as pure functions of facts the backend
// gathers, so they are testable on a host with no GPU
// (tests/test-sycl-dense-block-exec.cpp):
//   - dense_exec_first_failing_precheck: may the executor run this graph?
//     Split into the context gates and the block gates, whose verdict
//     dense_exec_block_memo keeps per placement plan. The composed form is
//     the reference ordering; the backend runs the two halves separately.
//   - dense_exec_build_plan: node devices, ranges, per-range copies and the
//     layout of the persistent per-device arena.
//   - dense_exec_graph_first_off: do the ranges of a decode graph record and
//     replay command graphs?
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace ggml_sycl {

// Listed in evaluation order. NONE means the executor ran the graph.
enum dense_exec_gate {
    DENSE_EXEC_GATE_NONE = 0,
    DENSE_EXEC_GATE_DISABLED,          // GGML_SYCL_BLOCK_EXEC_DENSE=0
    DENSE_EXEC_GATE_NO_GRAPH,          // no cgraph to execute
    DENSE_EXEC_GATE_GRAPH_RECORDING,   // a SYCL command graph is being recorded
    DENSE_EXEC_GATE_UNSUPPORTED_MODE,  // CPU offload or tensor parallelism is active
    DENSE_EXEC_GATE_NO_PLAN,           // no placement plan visible to the device
    DENSE_EXEC_GATE_FEW_BLOCKS,        // fewer than 2 active layer blocks
    DENSE_EXEC_GATE_NOT_DENSE,         // a block carries MoE expert weights, or the graph has a MUL_MAT_ID
    DENSE_EXEC_GATE_KV_DEVICE,         // a block's KV cache is not on its execution device
    DENSE_EXEC_GATE_SINGLE_DEVICE,     // every node of this graph runs on the backend's own device
    DENSE_EXEC_GATE_TOO_MANY_RANGES,   // the device assignment interleaves more than the plan allows
    DENSE_EXEC_GATE_OPERAND,           // a node reads an operand its device cannot reach
    DENSE_EXEC_GATE_IN_PLACE,          // a node writes into storage produced outside its range
    DENSE_EXEC_GATE_OUTPUT,            // a tensor produced on another device is a graph output
    DENSE_EXEC_GATE_ARENA_TOO_LARGE,   // a device arena exceeds GGML_SYCL_BLOCK_EXEC_DENSE_MAX_ARENA_MB
    DENSE_EXEC_GATE_PREPARE_FAILED,    // arena allocation or slicing failed
    // Copying the operands into a range failed before that range ran any
    // node. The rest of the graph ran on the per-op path instead.
    DENSE_EXEC_GATE_STAGE_FAILED,
};

inline const char * dense_exec_gate_name(dense_exec_gate gate) {
    switch (gate) {
        case DENSE_EXEC_GATE_NONE:
            return "none";
        case DENSE_EXEC_GATE_DISABLED:
            return "disabled";
        case DENSE_EXEC_GATE_NO_GRAPH:
            return "no-graph";
        case DENSE_EXEC_GATE_GRAPH_RECORDING:
            return "graph-recording";
        case DENSE_EXEC_GATE_UNSUPPORTED_MODE:
            return "unsupported-mode";
        case DENSE_EXEC_GATE_NO_PLAN:
            return "no-plan";
        case DENSE_EXEC_GATE_FEW_BLOCKS:
            return "few-blocks";
        case DENSE_EXEC_GATE_NOT_DENSE:
            return "not-dense";
        case DENSE_EXEC_GATE_KV_DEVICE:
            return "kv-device";
        case DENSE_EXEC_GATE_SINGLE_DEVICE:
            return "single-device";
        case DENSE_EXEC_GATE_TOO_MANY_RANGES:
            return "too-many-ranges";
        case DENSE_EXEC_GATE_OPERAND:
            return "operand";
        case DENSE_EXEC_GATE_IN_PLACE:
            return "in-place";
        case DENSE_EXEC_GATE_OUTPUT:
            return "output";
        case DENSE_EXEC_GATE_ARENA_TOO_LARGE:
            return "arena-too-large";
        case DENSE_EXEC_GATE_PREPARE_FAILED:
            return "prepare-failed";
        case DENSE_EXEC_GATE_STAGE_FAILED:
            return "stage-failed";
    }
    return "unknown";
}

// Parses GGML_SYCL_BLOCK_EXEC_DENSE, given its value or nullptr when unset.
// On by default; any value atoi reads as 0 is the opt-out, including an empty
// value and words such as "off" or "true". The per-graph gates below, not this
// variable, keep the executor off graphs it does not handle.
inline bool dense_exec_env_enabled(const char * env) {
    return env == nullptr || std::atoi(env) != 0;
}

// One active layer block of the placement plan.
struct dense_exec_block {
    int  start_layer      = -1;
    int  end_layer        = -1;
    int  execution_device = -1;
    int  kv_device        = -1;
    bool has_moe_weights  = false;
};

struct dense_exec_precheck_inputs {
    bool                          enabled          = false;
    bool                          has_graph        = false;
    bool                          graph_recording  = false;
    bool                          unsupported_mode = false;
    bool                          has_plan         = false;
    // Read only by the composed dense_exec_first_failing_precheck; the backend
    // takes the blocks from dense_exec_block_memo instead of filling this.
    std::vector<dense_exec_block> blocks;
};

// The first of the gates that need no plan blocks: the switch, the graph, the
// mode and whether a plan exists. DENSE_EXEC_GATE_NONE when all of them pass;
// `in.blocks` is not read.
inline dense_exec_gate dense_exec_first_failing_context_precheck(const dense_exec_precheck_inputs & in) {
    if (!in.enabled) {
        return DENSE_EXEC_GATE_DISABLED;
    }
    if (!in.has_graph) {
        return DENSE_EXEC_GATE_NO_GRAPH;
    }
    if (in.graph_recording) {
        return DENSE_EXEC_GATE_GRAPH_RECORDING;
    }
    if (in.unsupported_mode) {
        return DENSE_EXEC_GATE_UNSUPPORTED_MODE;
    }
    if (!in.has_plan) {
        return DENSE_EXEC_GATE_NO_PLAN;
    }
    return DENSE_EXEC_GATE_NONE;
}

// The first of the gates decided by the plan's layer blocks alone. They depend
// on nothing else, so the backend judges a plan once (dense_exec_block_memo).
inline dense_exec_gate dense_exec_first_failing_block_precheck(const std::vector<dense_exec_block> & blocks) {
    if (blocks.size() < 2) {
        return DENSE_EXEC_GATE_FEW_BLOCKS;
    }
    for (const dense_exec_block & block : blocks) {
        if (block.has_moe_weights) {
            return DENSE_EXEC_GATE_NOT_DENSE;
        }
    }
    for (const dense_exec_block & block : blocks) {
        if (block.execution_device < 0 || block.kv_device != block.execution_device) {
            return DENSE_EXEC_GATE_KV_DEVICE;
        }
    }
    return DENSE_EXEC_GATE_NONE;
}

// The first gate, among those decidable from the plan alone, that stops the
// executor; DENSE_EXEC_GATE_NONE when all of them pass. This is the reference
// ordering: the backend's prepare() mirrors it in two steps, the context
// gates per graph and then the block verdict from dense_exec_block_memo.
inline dense_exec_gate dense_exec_first_failing_precheck(const dense_exec_precheck_inputs & in) {
    const dense_exec_gate gate = dense_exec_first_failing_context_precheck(in);
    return gate != DENSE_EXEC_GATE_NONE ? gate : dense_exec_first_failing_block_precheck(in.blocks);
}

// The blocks of the last placement plan judged, and their verdict. Graphs the
// block gates reject (one card, MoE) are computed every token; with the memo
// they cost an identity check instead of rebuilding the blocks. The plan is
// held weakly, so a replan that frees it cannot alias its successor.
struct dense_exec_block_memo {
    std::weak_ptr<const void>     plan;
    std::vector<dense_exec_block> blocks;
    dense_exec_gate               gate = DENSE_EXEC_GATE_NONE;
};

// True when `memo` was judged for `plan`, which must be the live owner.
inline bool dense_exec_block_memo_current(const dense_exec_block_memo &       memo,
                                          const std::shared_ptr<const void> & plan) {
    return plan != nullptr && memo.plan.lock() == plan;
}

inline void dense_exec_block_memo_store(dense_exec_block_memo &             memo,
                                        const std::shared_ptr<const void> & plan,
                                        std::vector<dense_exec_block>       blocks) {
    memo.plan   = plan;
    memo.blocks = std::move(blocks);
    memo.gate   = dense_exec_first_failing_block_precheck(memo.blocks);
}

// ---------------------------------------------------------------------------
// Range command graphs
// ---------------------------------------------------------------------------

// Why the ranges of a graph the executor runs are dispatched directly instead
// of recorded once and replayed. Listed in evaluation order; the last two are
// decided after this check, by the backend. NONE means they record or replay.
enum dense_graph_off {
    DENSE_GRAPH_OFF_NONE = 0,
    DENSE_GRAPH_OFF_ENV,            // GGML_SYCL_BLOCK_EXEC_DENSE_GRAPH=0
    DENSE_GRAPH_OFF_DISABLE_GRAPH,  // GGML_SYCL_DISABLE_GRAPH
    // Diagnostics that wait on or read back from the queue inside the node
    // loop, which a recording queue refuses (the eval would fail).
    DENSE_GRAPH_OFF_SAFE_MODE,      // GGML_SYCL_SAFE_MODE (also implies DISABLE_GRAPH at init)
    DENSE_GRAPH_OFF_OP_TIMING,      // GGML_SYCL_OP_TIMING: a queue wait around every op
    DENSE_GRAPH_OFF_DEBUG_SYNC,     // GGML_SYCL_DEBUG_SYNC, _OPS or _NAMES: a wait after the matched ops
    DENSE_GRAPH_OFF_NAN_CHECK,      // GGML_SYCL_NAN_CHECK: a readback after every float op
    DENSE_GRAPH_OFF_TENSOR_TRACE,   // GGML_SYCL_TENSOR_TRACE, _TG_TRACE_HASH, _TG_DUMP_*: readbacks
    DENSE_GRAPH_OFF_MULTITHREADED,  // graphs computed from several threads
    DENSE_GRAPH_OFF_DISABLED,       // graphs disabled on the context, or a range recording failed
    DENSE_GRAPH_OFF_NOT_DECODE,     // only decode graphs record
    DENSE_GRAPH_OFF_HOST_INPUTS,    // GGML_SYCL_DISABLE_DECODE_GRAPH_HOST_INPUTS and the graph has host inputs
    DENSE_GRAPH_OFF_FA_UNVERIFIED,  // the whole-graph FA gate refuses
    DENSE_GRAPH_OFF_INCOMPATIBLE,   // check_graph_compatibility refuses
    DENSE_GRAPH_OFF_STAGE_FAILED,   // the executor fell back to the per-op path
    DENSE_GRAPH_OFF_LAST = DENSE_GRAPH_OFF_STAGE_FAILED,
};

inline const char * dense_exec_graph_off_name(dense_graph_off reason) {
    switch (reason) {
        case DENSE_GRAPH_OFF_NONE:
            return "none";
        case DENSE_GRAPH_OFF_ENV:
            return "env";
        case DENSE_GRAPH_OFF_DISABLE_GRAPH:
            return "disable-graph";
        case DENSE_GRAPH_OFF_SAFE_MODE:
            return "safe-mode";
        case DENSE_GRAPH_OFF_OP_TIMING:
            return "op-timing";
        case DENSE_GRAPH_OFF_DEBUG_SYNC:
            return "debug-sync";
        case DENSE_GRAPH_OFF_NAN_CHECK:
            return "nan-check";
        case DENSE_GRAPH_OFF_TENSOR_TRACE:
            return "tensor-trace";
        case DENSE_GRAPH_OFF_MULTITHREADED:
            return "multithreaded";
        case DENSE_GRAPH_OFF_DISABLED:
            return "disabled";
        case DENSE_GRAPH_OFF_NOT_DECODE:
            return "not-decode";
        case DENSE_GRAPH_OFF_HOST_INPUTS:
            return "host-inputs";
        case DENSE_GRAPH_OFF_FA_UNVERIFIED:
            return "fa-unverified";
        case DENSE_GRAPH_OFF_INCOMPATIBLE:
            return "incompatible";
        case DENSE_GRAPH_OFF_STAGE_FAILED:
            return "stage-failed";
    }
    return "unknown";
}

// GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW, as the whole-graph decode gate reads it.
enum dense_graph_fa_mode {
    DENSE_GRAPH_FA_AUTO = 0,
    DENSE_GRAPH_FA_FORCE_ON,
    DENSE_GRAPH_FA_FORCE_OFF,
};

struct dense_graph_facts {
    bool                enabled             = false;
    bool                disable_graph       = false;
    bool                safe_mode           = false;
    bool                op_timing           = false;
    bool                debug_sync          = false;
    bool                nan_check           = false;
    bool                tensor_trace        = false;
    bool                multithreaded       = false;
    bool                disabled            = false;
    bool                is_decode           = false;
    bool                host_inputs_blocked = false;
    bool                has_fa              = false;
    dense_graph_fa_mode fa_mode             = DENSE_GRAPH_FA_AUTO;
    // Every decode FA dispatch the context has observed, on any device,
    // reached a kernel verified replay-safe (and at least one was observed).
    bool                fa_observed_safe    = false;
    bool                fa_mask_sinks_safe  = false;
};

// The first reason, among those decidable before recording, that keeps the
// ranges on direct dispatch; DENSE_GRAPH_OFF_NONE when none applies.
inline dense_graph_off dense_exec_graph_first_off(const dense_graph_facts & f) {
    if (!f.enabled) {
        return DENSE_GRAPH_OFF_ENV;
    }
    if (f.disable_graph) {
        return DENSE_GRAPH_OFF_DISABLE_GRAPH;
    }
    if (f.safe_mode) {
        return DENSE_GRAPH_OFF_SAFE_MODE;
    }
    if (f.op_timing) {
        return DENSE_GRAPH_OFF_OP_TIMING;
    }
    if (f.debug_sync) {
        return DENSE_GRAPH_OFF_DEBUG_SYNC;
    }
    if (f.nan_check) {
        return DENSE_GRAPH_OFF_NAN_CHECK;
    }
    if (f.tensor_trace) {
        return DENSE_GRAPH_OFF_TENSOR_TRACE;
    }
    if (f.multithreaded) {
        return DENSE_GRAPH_OFF_MULTITHREADED;
    }
    if (f.disabled) {
        return DENSE_GRAPH_OFF_DISABLED;
    }
    if (!f.is_decode) {
        return DENSE_GRAPH_OFF_NOT_DECODE;
    }
    if (f.host_inputs_blocked) {
        return DENSE_GRAPH_OFF_HOST_INPUTS;
    }
    if (f.has_fa) {
        const bool engage = f.fa_mode == DENSE_GRAPH_FA_FORCE_ON ||
                            (f.fa_mode == DENSE_GRAPH_FA_AUTO && f.fa_observed_safe && f.fa_mask_sinks_safe);
        if (!engage) {
            return DENSE_GRAPH_OFF_FA_UNVERIFIED;
        }
    }
    return DENSE_GRAPH_OFF_NONE;
}

// ---------------------------------------------------------------------------
// Graph facts
// ---------------------------------------------------------------------------

// Storage is tracked by root: a view reads and writes its view_src chain's
// root, so every edge below names the root it touches, never a view.
enum dense_exec_root_kind {
    DENSE_EXEC_ROOT_NODE,     // produced by a node of this graph
    DENSE_EXEC_ROOT_WEIGHT,   // a model weight leaf
    DENSE_EXEC_ROOT_CONTROL,  // a per-graph host input leaf (positions, masks, row ids)
    DENSE_EXEC_ROOT_STATE,    // any other leaf: the KV cache, recurrent state
};

struct dense_exec_root {
    dense_exec_root_kind kind          = DENSE_EXEC_ROOT_NODE;
    int                  producer      = -1;  // NODE: index of the node that produces it
    size_t               bytes         = 0;
    bool                 is_output     = false;
    // Leaves: bit d is set when device d reads the leaf in place as a device
    // operand. Placement put it there; the executor never moves a weight or
    // the KV cache.
    uint32_t             resident_mask = 0;
};

struct dense_exec_node {
    // Layer index from the node's own name ("attn_norm-30"), -1 when the name
    // carries none. A source's name is deliberately not consulted: the final
    // "norm" would inherit the last layer's index from its source, although the
    // output norm weight lives on another card.
    int              own_layer = -1;
    // Views, reshapes, permutes and transposes: no kernel runs.
    bool             is_noop   = false;
    // Root this node writes; -1 for a no-op. A node that writes into a view
    // (SET_ROWS into the KV cache, an in-place op) names that view's root.
    int              dst_root  = -1;
    std::vector<int> src_roots;
};

struct dense_exec_graph {
    std::vector<dense_exec_node>  nodes;
    std::vector<dense_exec_root>  roots;
    std::vector<dense_exec_block> blocks;
    int                           original_device = 0;  // the backend context's device
    size_t                        max_ranges      = 0;  // 0: 2 * blocks + 1
    size_t                        max_arena_bytes = 0;  // per device; 0: unlimited
};

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

constexpr size_t dense_exec_slice_alignment = 256;
constexpr int    dense_exec_max_devices     = 32;

// Nodes [begin, end) on `device`. A range on the original device runs exactly
// as a graph without the executor does; any other range is an executor range.
struct dense_exec_range {
    int  begin    = 0;
    int  end      = 0;
    int  device   = -1;
    bool executor = false;
};

// Storage for one root on one device, inside that device's persistent arena.
struct dense_exec_slice {
    int    root   = -1;
    int    device = -1;
    size_t offset = 0;
    size_t bytes  = 0;
};

struct dense_exec_copy {
    int from_slice = -1;
    int to_slice   = -1;
};

struct dense_exec_range_io {
    // Executor ranges only. Slices filled before the range runs, each copied
    // from the root's storage on the original device: for a NODE root, where
    // its producer there wrote it; for a CONTROL root, the device copy the
    // host inputs were uploaded into.
    std::vector<int>             stage_in;
    // Slices published as their root's storage on the range's device while
    // the range runs, and unpublished when it ends.
    std::vector<int>             publish;
    // After the range: copies of roots it produced that a later range on the
    // original device reads. Each destination slice stays published as its
    // root's storage on the original device until the graph ends.
    std::vector<dense_exec_copy> copy_out;
    // Byte offsets, parallel to stage_in and copy_out, of each copy's span in
    // the host staging buffer of either device it crosses between. Each
    // direction packs its copies into one buffer so they are submitted
    // together and waited on once.
    std::vector<size_t>          stage_in_host;
    std::vector<size_t>          copy_out_host;
};

struct dense_exec_plan {
    std::vector<int>                 node_device;
    std::vector<dense_exec_range>    ranges;
    std::vector<dense_exec_range_io> io;           // one per range
    std::vector<dense_exec_slice>    slices;
    std::vector<size_t>              arena_bytes;  // indexed by device
    size_t                           host_stage_bytes = 0;  // the widest direction of any range
    int                              failing_node     = -1;
};

inline size_t dense_exec_align(size_t bytes) {
    return (bytes + dense_exec_slice_alignment - 1) / dense_exec_slice_alignment * dense_exec_slice_alignment;
}

inline int dense_exec_block_device(const std::vector<dense_exec_block> & blocks, int layer) {
    if (layer < 0) {
        return -1;
    }
    for (const dense_exec_block & block : blocks) {
        if (layer >= block.start_layer && layer <= block.end_layer) {
            return block.execution_device;
        }
    }
    return -1;
}

// The device each node runs on.
//   1. A no-op runs nowhere; it takes its predecessor's device so it never
//      splits a range.
//   2. Otherwise the preferred device is the node's own layer's block, else
//      the device of its first source produced by another node, else the
//      original device.
//   3. Placement then decides: when the node reads weights, or writes into a
//      leaf such as the KV cache, it runs where all of those reside -- the
//      preferred device when they reside there too. Host-resident weights run
//      on the original device, whose per-op path already serves them.
inline std::vector<int> dense_exec_assign_devices(const dense_exec_graph & g) {
    std::vector<int> node_device(g.nodes.size(), g.original_device);
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const dense_exec_node & node = g.nodes[i];
        if (node.is_noop) {
            node_device[i] = i > 0 ? node_device[i - 1] : g.original_device;
            continue;
        }

        int preferred = dense_exec_block_device(g.blocks, node.own_layer);
        if (preferred < 0) {
            for (int r : node.src_roots) {
                const dense_exec_root & root = g.roots[static_cast<size_t>(r)];
                if (root.kind == DENSE_EXEC_ROOT_NODE && root.producer >= 0 && static_cast<size_t>(root.producer) < i) {
                    preferred = node_device[static_cast<size_t>(root.producer)];
                    break;
                }
            }
        }
        if (preferred < 0) {
            preferred = g.original_device;
        }

        bool     placed      = false;
        uint32_t placed_mask = ~uint32_t{ 0 };
        for (int r : node.src_roots) {
            const dense_exec_root & root = g.roots[static_cast<size_t>(r)];
            if (root.kind == DENSE_EXEC_ROOT_WEIGHT) {
                placed = true;
                placed_mask &= root.resident_mask;
            }
        }
        if (node.dst_root >= 0) {
            const dense_exec_root & root = g.roots[static_cast<size_t>(node.dst_root)];
            if (root.kind == DENSE_EXEC_ROOT_STATE) {
                placed = true;
                placed_mask &= root.resident_mask;
            }
        }

        int device = preferred;
        if (placed) {
            if (preferred >= dense_exec_max_devices || (placed_mask & (uint32_t{ 1 } << preferred)) == 0) {
                device = g.original_device;
                for (int d = 0; d < dense_exec_max_devices; ++d) {
                    if (placed_mask & (uint32_t{ 1 } << d)) {
                        device = d;
                        break;
                    }
                }
            }
        }
        node_device[i] = device;
    }
    return node_device;
}

inline std::vector<dense_exec_range> dense_exec_build_ranges(const std::vector<int> & node_device,
                                                             int                      original_device) {
    std::vector<dense_exec_range> ranges;
    for (size_t i = 0; i < node_device.size(); ++i) {
        if (ranges.empty() || ranges.back().device != node_device[i]) {
            dense_exec_range range{};
            range.begin    = static_cast<int>(i);
            range.device   = node_device[i];
            range.executor = node_device[i] != original_device;
            ranges.push_back(range);
        }
        ranges.back().end = static_cast<int>(i) + 1;
    }
    return ranges;
}

// Builds the whole plan, or returns the gate that rejects the graph (with
// out.failing_node naming the node when one is to blame).
inline dense_exec_gate dense_exec_build_plan(const dense_exec_graph & g, dense_exec_plan & out) {
    out = dense_exec_plan{};
    if (g.original_device < 0 || g.original_device >= dense_exec_max_devices) {
        return DENSE_EXEC_GATE_SINGLE_DEVICE;
    }

    out.node_device = dense_exec_assign_devices(g);
    out.ranges      = dense_exec_build_ranges(out.node_device, g.original_device);

    bool any_executor = false;
    for (const dense_exec_range & range : out.ranges) {
        if (range.executor) {
            if (range.device < 0 || range.device >= dense_exec_max_devices) {
                out.failing_node = range.begin;
                return DENSE_EXEC_GATE_OPERAND;
            }
            any_executor = true;
        }
    }
    if (!any_executor) {
        return DENSE_EXEC_GATE_SINGLE_DEVICE;
    }
    const size_t max_ranges = g.max_ranges != 0 ? g.max_ranges : 2 * g.blocks.size() + 1;
    if (out.ranges.size() > max_ranges) {
        return DENSE_EXEC_GATE_TOO_MANY_RANGES;
    }

    std::vector<int> node_range(g.nodes.size(), -1);
    for (size_t r = 0; r < out.ranges.size(); ++r) {
        for (int i = out.ranges[r].begin; i < out.ranges[r].end; ++i) {
            node_range[static_cast<size_t>(i)] = static_cast<int>(r);
        }
    }
    auto range_of_root = [&](const dense_exec_root & root) {
        return root.producer >= 0 && static_cast<size_t>(root.producer) < node_range.size() ?
                   node_range[static_cast<size_t>(root.producer)] :
                   -1;
    };

    // slice_of[root * devices + device]: index into out.slices, -1 if none.
    std::vector<int> slice_of(g.roots.size() * dense_exec_max_devices, -1);
    auto             ensure_slice = [&](int root, int device) {
        int & idx = slice_of[static_cast<size_t>(root) * dense_exec_max_devices + static_cast<size_t>(device)];
        if (idx < 0) {
            dense_exec_slice slice{};
            slice.root   = root;
            slice.device = device;
            slice.bytes  = g.roots[static_cast<size_t>(root)].bytes;
            idx          = static_cast<int>(out.slices.size());
            out.slices.push_back(slice);
        }
        return idx;
    };
    auto push_unique = [](std::vector<int> & v, int x) {
        for (int y : v) {
            if (y == x) {
                return;
            }
        }
        v.push_back(x);
    };

    out.io.resize(out.ranges.size());
    for (size_t r = 0; r < out.ranges.size(); ++r) {
        const dense_exec_range & range = out.ranges[r];
        if (!range.executor) {
            continue;
        }
        const int             d    = range.device;
        const uint32_t        bit  = uint32_t{ 1 } << d;
        dense_exec_range_io & io   = out.io[r];
        auto                  fail = [&](int node, dense_exec_gate gate) {
            out.failing_node = node;
            return gate;
        };

        for (int i = range.begin; i < range.end; ++i) {
            const dense_exec_node & node = g.nodes[static_cast<size_t>(i)];
            if (node.is_noop) {
                continue;
            }

            for (int s : node.src_roots) {
                const dense_exec_root & root = g.roots[static_cast<size_t>(s)];
                switch (root.kind) {
                    case DENSE_EXEC_ROOT_NODE:
                        {
                            const int pr = range_of_root(root);
                            if (pr == static_cast<int>(r)) {
                                break;  // produced earlier in this range: already published
                            }
                            if (pr < 0) {
                                return fail(i, DENSE_EXEC_GATE_OPERAND);
                            }
                            const dense_exec_range & producer_range = out.ranges[static_cast<size_t>(pr)];
                            if (!producer_range.executor) {
                                // Re-staged for every executor range: a later
                                // original-device node may have rewritten it.
                                const int slice = ensure_slice(s, d);
                                push_unique(io.stage_in, slice);
                                push_unique(io.publish, slice);
                            } else if (producer_range.device == d) {
                                push_unique(io.publish, ensure_slice(s, d));
                            } else {
                                return fail(i, DENSE_EXEC_GATE_OPERAND);
                            }
                            break;
                        }
                    case DENSE_EXEC_ROOT_CONTROL:
                        if ((root.resident_mask & bit) == 0) {
                            const int slice = ensure_slice(s, d);
                            push_unique(io.stage_in, slice);
                            push_unique(io.publish, slice);
                        }
                        break;
                    case DENSE_EXEC_ROOT_WEIGHT:
                    case DENSE_EXEC_ROOT_STATE:
                        if ((root.resident_mask & bit) == 0) {
                            return fail(i, DENSE_EXEC_GATE_OPERAND);
                        }
                        break;
                }
            }

            if (node.dst_root >= 0) {
                const dense_exec_root & root = g.roots[static_cast<size_t>(node.dst_root)];
                if (root.kind == DENSE_EXEC_ROOT_NODE && range_of_root(root) == static_cast<int>(r)) {
                    if (root.is_output) {
                        return fail(i, DENSE_EXEC_GATE_OUTPUT);
                    }
                    push_unique(io.publish, ensure_slice(node.dst_root, d));
                } else if (root.kind == DENSE_EXEC_ROOT_STATE && (root.resident_mask & bit) != 0) {
                    // Written in place where placement put it.
                } else {
                    return fail(i, DENSE_EXEC_GATE_IN_PLACE);
                }
            }
        }
    }

    // Copies back to the original device: every root an executor range
    // produced that a node of a later original-device range reads or writes.
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const int cr = node_range[i];
        if (out.ranges[static_cast<size_t>(cr)].executor || g.nodes[i].is_noop) {
            continue;
        }
        // Writing such a root in place on the original device would leave
        // the executor device's slice stale: a later range on that device
        // re-publishes the slice without re-staging it.
        if (g.nodes[i].dst_root >= 0) {
            const dense_exec_root & dst = g.roots[static_cast<size_t>(g.nodes[i].dst_root)];
            const int               pr  = dst.kind == DENSE_EXEC_ROOT_NODE ? range_of_root(dst) : -1;
            if (pr >= 0 && out.ranges[static_cast<size_t>(pr)].executor) {
                out.failing_node = static_cast<int>(i);
                return DENSE_EXEC_GATE_IN_PLACE;
            }
        }
        std::vector<int> touched = g.nodes[i].src_roots;
        if (g.nodes[i].dst_root >= 0) {
            touched.push_back(g.nodes[i].dst_root);
        }
        for (int s : touched) {
            const dense_exec_root & root = g.roots[static_cast<size_t>(s)];
            if (root.kind != DENSE_EXEC_ROOT_NODE) {
                continue;
            }
            const int pr = range_of_root(root);
            if (pr < 0 || pr >= cr || !out.ranges[static_cast<size_t>(pr)].executor) {
                continue;
            }
            const int from = ensure_slice(s, out.ranges[static_cast<size_t>(pr)].device);
            const int to   = ensure_slice(s, g.original_device);
            bool      seen = false;
            for (const dense_exec_copy & c : out.io[static_cast<size_t>(pr)].copy_out) {
                seen = seen || c.to_slice == to;
            }
            if (!seen) {
                out.io[static_cast<size_t>(pr)].copy_out.push_back(dense_exec_copy{ from, to });
            }
        }
    }

    out.arena_bytes.assign(dense_exec_max_devices, 0);
    for (dense_exec_slice & slice : out.slices) {
        size_t & total = out.arena_bytes[static_cast<size_t>(slice.device)];
        slice.offset   = total;
        total += dense_exec_align(slice.bytes != 0 ? slice.bytes : 1);
    }
    out.host_stage_bytes = 0;
    for (dense_exec_range_io & io : out.io) {
        size_t in_bytes = 0;
        for (int s : io.stage_in) {
            io.stage_in_host.push_back(in_bytes);
            in_bytes += dense_exec_align(out.slices[static_cast<size_t>(s)].bytes);
        }
        size_t out_bytes = 0;
        for (const dense_exec_copy & c : io.copy_out) {
            io.copy_out_host.push_back(out_bytes);
            out_bytes += dense_exec_align(out.slices[static_cast<size_t>(c.from_slice)].bytes);
        }
        out.host_stage_bytes = std::max(out.host_stage_bytes, std::max(in_bytes, out_bytes));
    }
    if (g.max_arena_bytes != 0) {
        for (size_t total : out.arena_bytes) {
            if (total > g.max_arena_bytes) {
                return DENSE_EXEC_GATE_ARENA_TOO_LARGE;
            }
        }
    }
    return DENSE_EXEC_GATE_NONE;
}

// The ownership invariants the runner relies on when it publishes and
// restores slices; nullptr when the plan keeps all of them, else the first
// one it breaks.
//   - one slice per (root, device), so two publications of one root on one
//     device can never name different storage;
//   - an executor range publishes only slices on its own device, at most one
//     per root; a staged input is among them;
//   - a staged input is never a root the same range produces: a source
//     override may not sit on the root of a result the range publishes
//     (the llama.cpp-kw7x clobber);
//   - a copy out goes from the producing range's device to the original
//     device, between two slices of the same root;
//   - no original-device node writes a root an executor range produced, which
//     would leave that range device's slice stale for a later range there.
inline const char * dense_exec_plan_violation(const dense_exec_graph & g, const dense_exec_plan & p) {
    for (size_t a = 0; a < p.slices.size(); ++a) {
        for (size_t b = a + 1; b < p.slices.size(); ++b) {
            if (p.slices[a].root == p.slices[b].root && p.slices[a].device == p.slices[b].device) {
                return "two slices for one root on one device";
            }
        }
    }
    if (p.io.size() != p.ranges.size()) {
        return "io does not match ranges";
    }
    for (size_t r = 0; r < p.ranges.size(); ++r) {
        const dense_exec_range &    range = p.ranges[r];
        const dense_exec_range_io & io    = p.io[r];
        if (!range.executor) {
            if (!io.stage_in.empty() || !io.publish.empty() || !io.copy_out.empty()) {
                return "an original-device range carries io";
            }
            continue;
        }
        for (size_t a = 0; a < io.publish.size(); ++a) {
            const dense_exec_slice & sa = p.slices[static_cast<size_t>(io.publish[a])];
            if (sa.device != range.device) {
                return "a range publishes a slice of another device";
            }
            for (size_t b = a + 1; b < io.publish.size(); ++b) {
                if (sa.root == p.slices[static_cast<size_t>(io.publish[b])].root) {
                    return "a range publishes one root twice";
                }
            }
        }
        for (int s : io.stage_in) {
            bool published = false;
            for (int q : io.publish) {
                published = published || q == s;
            }
            if (!published) {
                return "a staged slice is not published";
            }
            const dense_exec_root & root = g.roots[static_cast<size_t>(p.slices[static_cast<size_t>(s)].root)];
            if (root.kind == DENSE_EXEC_ROOT_NODE && root.producer >= range.begin && root.producer < range.end) {
                return "a range stages over a root it produces";
            }
        }
        for (const dense_exec_copy & c : io.copy_out) {
            const dense_exec_slice & from = p.slices[static_cast<size_t>(c.from_slice)];
            const dense_exec_slice & to   = p.slices[static_cast<size_t>(c.to_slice)];
            if (from.root != to.root || from.device != range.device || to.device != g.original_device) {
                return "a copy out does not go from the range's device to the original device";
            }
        }
    }
    auto range_of_node = [&](int node) -> const dense_exec_range * {
        for (const dense_exec_range & range : p.ranges) {
            if (node >= range.begin && node < range.end) {
                return &range;
            }
        }
        return nullptr;
    };
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const dense_exec_node &  node  = g.nodes[i];
        const dense_exec_range * range = range_of_node(static_cast<int>(i));
        if (node.is_noop || node.dst_root < 0 || range == nullptr || range->executor) {
            continue;
        }
        const dense_exec_root & dst = g.roots[static_cast<size_t>(node.dst_root)];
        if (dst.kind != DENSE_EXEC_ROOT_NODE) {
            continue;
        }
        const dense_exec_range * producer = range_of_node(dst.producer);
        if (producer != nullptr && producer->executor) {
            return "an original-device node writes an executor range's result";
        }
    }

    // Each direction's copies land in disjoint spans inside the host buffer.
    auto spans_fit = [&](std::vector<std::pair<size_t, size_t>> spans) {  // (offset, bytes)
        std::sort(spans.begin(), spans.end());
        size_t end = 0;
        for (const auto & span : spans) {
            if (span.second == 0) {
                continue;
            }
            if (span.first < end || span.first + span.second > p.host_stage_bytes) {
                return false;
            }
            end = span.first + span.second;
        }
        return true;
    };
    for (const dense_exec_range_io & io : p.io) {
        if (io.stage_in_host.size() != io.stage_in.size() || io.copy_out_host.size() != io.copy_out.size()) {
            return "host staging spans overlap or leave the buffer";
        }
        std::vector<std::pair<size_t, size_t>> in, out;
        for (size_t k = 0; k < io.stage_in.size(); ++k) {
            in.emplace_back(io.stage_in_host[k], p.slices[static_cast<size_t>(io.stage_in[k])].bytes);
        }
        for (size_t k = 0; k < io.copy_out.size(); ++k) {
            out.emplace_back(io.copy_out_host[k], p.slices[static_cast<size_t>(io.copy_out[k].from_slice)].bytes);
        }
        if (!spans_fit(std::move(in)) || !spans_fit(std::move(out))) {
            return "host staging spans overlap or leave the buffer";
        }
    }
    return nullptr;
}

// A pool frees scratch that a recording used into its own retained list, not
// into the recording's sink. When a range graph finishes recording, the
// entries appended since its recording began are moved into that graph, so the
// graph owns the scratch it baked. What the pool held before the baseline
// stays. A pool drained since the baseline gives up everything it now holds.
// Returns the number of entries moved.
template <typename T> inline size_t dense_exec_take_since(std::vector<T> & from, size_t baseline, std::vector<T> & to) {
    const size_t first = baseline <= from.size() ? baseline : 0;
    const size_t moved = from.size() - first;
    to.insert(to.end(), std::make_move_iterator(from.begin() + static_cast<std::ptrdiff_t>(first)),
              std::make_move_iterator(from.end()));
    from.resize(first);
    return moved;
}

}  // namespace ggml_sycl
