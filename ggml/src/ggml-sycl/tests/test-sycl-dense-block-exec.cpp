// Dense-split layer-block executor planning (llama.cpp-tf8m).
//
// Host-only: no SYCL queue, no device, no model. The planner is a pure
// function of graph facts the backend gathers, which is what makes it
// testable. The graphs below are shaped like a Mistral decode graph split over
// two cards: layers on device 0, then layers on device 1, then the output head
// back on device 0 because the output norm and output weights live there.

#include "block-exec-dense.hpp"
#include "kv-runtime-demotion.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml_sycl;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

namespace {

// Builds a dense_exec_graph node by node and remembers node/root indices by
// name so assertions can talk about "norm" instead of index 41.
struct graph_builder {
    dense_exec_graph         g;
    std::vector<std::string> node_names;
    std::vector<std::string> root_names;

    int root(const std::string & name, dense_exec_root_kind kind, uint32_t resident_mask = 0, size_t bytes = 1024) {
        dense_exec_root r{};
        r.kind          = kind;
        r.resident_mask = resident_mask;
        r.bytes         = bytes;
        g.roots.push_back(r);
        root_names.push_back(name);
        return static_cast<int>(g.roots.size()) - 1;
    }

    // A node producing its own root.
    int op(const std::string & name, int layer, std::vector<int> srcs, size_t bytes = 1024) {
        const int r                              = root(name, DENSE_EXEC_ROOT_NODE, 0, bytes);
        g.roots[static_cast<size_t>(r)].producer = static_cast<int>(g.nodes.size());
        dense_exec_node n{};
        n.own_layer = layer;
        n.dst_root  = r;
        n.src_roots = std::move(srcs);
        g.nodes.push_back(n);
        node_names.push_back(name);
        return r;
    }

    // A node writing into an existing root (SET_ROWS into the KV cache).
    void write_into(const std::string & name, int layer, int dst_root, std::vector<int> srcs) {
        dense_exec_node n{};
        n.own_layer = layer;
        n.dst_root  = dst_root;
        n.src_roots = std::move(srcs);
        g.nodes.push_back(n);
        node_names.push_back(name);
    }

    void noop(const std::string & name) {
        dense_exec_node n{};
        n.is_noop = true;
        g.nodes.push_back(n);
        node_names.push_back(name);
    }

    int node(const std::string & name) const {
        for (size_t i = 0; i < node_names.size(); ++i) {
            if (node_names[i] == name) {
                return static_cast<int>(i);
            }
        }
        throw std::runtime_error("no node " + name);
    }

    int root_index(const std::string & name) const {
        for (size_t i = 0; i < root_names.size(); ++i) {
            if (root_names[i] == name) {
                return static_cast<int>(i);
            }
        }
        throw std::runtime_error("no root " + name);
    }
};

constexpr uint32_t DEV0 = 1u << 0;
constexpr uint32_t DEV1 = 1u << 1;

struct split_options {
    int      layers            = 4;
    int      first_dev1_layer  = 2;
    uint32_t output_norm_mask  = DEV0;
    uint32_t output_mask       = DEV0;
    uint32_t kv_mask_for_dev1  = DEV1;
    bool     host_wq_on_layer2 = false;
};

// Mistral-shaped decode graph, `layers` layers, split at first_dev1_layer.
static graph_builder make_split_graph(const split_options & o = split_options{}) {
    graph_builder b;
    b.g.original_device = 0;
    b.g.blocks.push_back(dense_exec_block{ 0, o.first_dev1_layer - 1, 0, 0, false });
    b.g.blocks.push_back(dense_exec_block{ o.first_dev1_layer, o.layers - 1, 1, 1, false });

    const int tok_embd = b.root("token_embd.weight", DENSE_EXEC_ROOT_WEIGHT, DEV0);
    const int tokens   = b.root("inp_tokens", DENSE_EXEC_ROOT_CONTROL);
    const int pos      = b.root("inp_pos", DENSE_EXEC_ROOT_CONTROL);
    const int mask     = b.root("kq_mask", DENSE_EXEC_ROOT_CONTROL);
    const int kv_idx   = b.root("kv_idxs", DENSE_EXEC_ROOT_CONTROL);

    int cur = b.op("inp_embd", -1, { tok_embd, tokens });
    for (int l = 0; l < o.layers; ++l) {
        const std::string L      = "-" + std::to_string(l);
        const uint32_t    wmask  = l < o.first_dev1_layer ? DEV0 : DEV1;
        const uint32_t    kvmask = l < o.first_dev1_layer ? DEV0 : o.kv_mask_for_dev1;
        const int         w_norm = b.root("attn_norm.weight" + L, DENSE_EXEC_ROOT_WEIGHT, wmask);
        const int         wq =
            b.root("attn_q.weight" + L, DENSE_EXEC_ROOT_WEIGHT, (o.host_wq_on_layer2 && l == 2) ? 0u : wmask);
        const int wo   = b.root("attn_output.weight" + L, DENSE_EXEC_ROOT_WEIGHT, wmask);
        const int w_up = b.root("ffn_up.weight" + L, DENSE_EXEC_ROOT_WEIGHT, wmask);
        const int kv   = b.root("cache_k" + L, DENSE_EXEC_ROOT_STATE, kvmask);

        const int norm  = b.op("norm" + L, l, { cur });
        const int anorm = b.op("attn_norm" + L, l, { norm, w_norm });
        const int q     = b.op("Qcur" + L, l, { anorm, wq });
        b.noop("Qcur" + L + " (reshaped)");
        const int qr = b.op("Qcur_rope" + L, l, { q, pos });
        b.write_into("k_set_rows" + L, -1, kv, { qr, kv_idx });
        b.noop("cache_k" + L + " (view)");
        const int fa  = b.op("fattn" + L, l, { qr, kv, mask });
        const int out = b.op("attn_out" + L, l, { fa, wo });
        const int inp = b.op("ffn_inp" + L, l, { out, cur });
        const int up  = b.op("ffn_out" + L, l, { inp, w_up });
        cur           = b.op("l_out" + L, l, { up, inp });
    }
    const int w_onorm = b.root("output_norm.weight", DENSE_EXEC_ROOT_WEIGHT, o.output_norm_mask);
    const int w_out   = b.root("output.weight", DENSE_EXEC_ROOT_WEIGHT, o.output_mask);
    const int norm    = b.op("norm", -1, { cur });
    const int rnorm   = b.op("result_norm", -1, { norm, w_onorm });
    const int logits  = b.op("result_output", -1, { rnorm, w_out });
    b.g.roots[static_cast<size_t>(logits)].is_output = true;
    return b;
}

static bool contains(const std::vector<int> & v, int x) {
    for (int y : v) {
        if (y == x) {
            return true;
        }
    }
    return false;
}

static int slice_for(const dense_exec_plan & p, int root, int device) {
    for (size_t i = 0; i < p.slices.size(); ++i) {
        if (p.slices[i].root == root && p.slices[i].device == device) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

}  // namespace

static void test_precheck_gates() {
    dense_exec_precheck_inputs in{};
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_DISABLED, "disabled is the first gate");
    in.enabled = true;
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_NO_GRAPH, "no graph");
    in.has_graph       = true;
    in.graph_recording = true;
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_GRAPH_RECORDING, "recording");
    in.graph_recording  = false;
    in.unsupported_mode = true;
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_UNSUPPORTED_MODE, "cpu offload / TP");
    in.unsupported_mode = false;
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_NO_PLAN, "no plan");
    in.has_plan = true;
    in.blocks.push_back(dense_exec_block{ 0, 29, 0, 0, false });
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_FEW_BLOCKS, "one block is not a split");
    in.blocks.push_back(dense_exec_block{ 30, 31, 1, 1, true });
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_NOT_DENSE, "MoE blocks belong to another path");
    in.blocks[1].has_moe_weights = false;
    in.blocks[1].kv_device       = -1;
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_KV_DEVICE, "host KV is refused");
    in.blocks[1].kv_device = 0;
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_KV_DEVICE, "KV on another device is refused");
    in.blocks[1].kv_device = 1;
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_NONE, "a dense two-card split passes");

    for (int g = DENSE_EXEC_GATE_NONE; g <= DENSE_EXEC_GATE_STAGE_FAILED; ++g) {
        const char * name = dense_exec_gate_name(static_cast<dense_exec_gate>(g));
        check(name != nullptr && std::strcmp(name, "unknown") != 0, "every gate has a name: " + std::to_string(g));
        for (int h = DENSE_EXEC_GATE_NONE; h < g; ++h) {
            check(std::strcmp(name, dense_exec_gate_name(static_cast<dense_exec_gate>(h))) != 0,
                  "gate names are distinct: " + std::to_string(g));
        }
    }
}

// Whether a graph's ranges record or replay command graphs, and the first
// reason when they do not. The FA gate is the whole-graph decode gate.
static void test_graph_off_reasons() {
    dense_graph_facts f{};
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_ENV, "env is the first reason");
    f.enabled       = true;
    f.disable_graph = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_DISABLE_GRAPH, "GGML_SYCL_DISABLE_GRAPH");
    f.disable_graph = false;
    f.multithreaded = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_MULTITHREADED, "multithreaded compute");
    f.multithreaded = false;
    f.disabled      = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_DISABLED, "graphs disabled on the context");
    f.disabled = false;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_NOT_DECODE, "prefill never records");
    f.is_decode           = true;
    f.host_inputs_blocked = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_HOST_INPUTS, "host inputs blocked by env");
    f.host_inputs_blocked = false;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_NONE, "a decode graph without FA records");

    f.has_fa = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_FA_UNVERIFIED, "FA with nothing observed");
    f.fa_observed_safe = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_FA_UNVERIFIED, "FA whose mask/sinks cannot refresh");
    f.fa_mask_sinks_safe = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_NONE, "FA observed safe and refreshable");
    f.fa_mode = DENSE_GRAPH_FA_FORCE_OFF;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_FA_UNVERIFIED, "force-off wins over observation");
    f.fa_mode            = DENSE_GRAPH_FA_FORCE_ON;
    f.fa_observed_safe   = false;
    f.fa_mask_sinks_safe = false;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_NONE, "force-on skips the observation");
    f.is_decode = false;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_NOT_DECODE, "force-on does not reach prefill");

    for (int r = DENSE_GRAPH_OFF_NONE; r <= DENSE_GRAPH_OFF_LAST; ++r) {
        const char * name = dense_exec_graph_off_name(static_cast<dense_graph_off>(r));
        check(name != nullptr && std::strcmp(name, "unknown") != 0, "every reason has a name: " + std::to_string(r));
        for (int q = DENSE_GRAPH_OFF_NONE; q < r; ++q) {
            check(std::strcmp(name, dense_exec_graph_off_name(static_cast<dense_graph_off>(q))) != 0,
                  "reason names are distinct: " + std::to_string(r));
        }
    }
}

// A diagnostic env that waits on or reads back from the queue inside the node
// loop cannot run while that loop is being recorded: the eval would fail. Each
// keeps the ranges on direct dispatch, ahead of every graph-shape reason.
static void test_graph_off_debug_envs() {
    auto decode = [] {
        dense_graph_facts f{};
        f.enabled   = true;
        f.is_decode = true;
        return f;
    };
    check(dense_exec_graph_first_off(decode()) == DENSE_GRAPH_OFF_NONE, "the base facts record");

    dense_graph_facts f = decode();
    f.safe_mode         = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_SAFE_MODE, "GGML_SYCL_SAFE_MODE");
    f.disable_graph = true;  // what SAFE_MODE sets at init; the older reason names it
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_DISABLE_GRAPH, "SAFE_MODE's implied disable_graph");

    f           = decode();
    f.op_timing = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_OP_TIMING, "GGML_SYCL_OP_TIMING");

    f            = decode();
    f.debug_sync = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_DEBUG_SYNC, "GGML_SYCL_DEBUG_SYNC(_OPS|_NAMES)");

    f           = decode();
    f.nan_check = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_NAN_CHECK, "GGML_SYCL_NAN_CHECK");

    f              = decode();
    f.tensor_trace = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_TENSOR_TRACE, "tensor readback traces");

    // Ahead of the reasons that depend on the graph: a diagnostic run names
    // the diagnostic even on prefill or with the context's graphs disabled.
    f               = decode();
    f.op_timing     = true;
    f.is_decode     = false;
    f.disabled      = true;
    f.multithreaded = true;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_OP_TIMING, "a diagnostic outranks the graph reasons");
    f.enabled = false;
    check(dense_exec_graph_first_off(f) == DENSE_GRAPH_OFF_ENV, "the kill switch still comes first");
}

// The case the executor exists for: three ranges, layers 2-3 plus the final
// norm on device 1, the output head back on device 0.
static void test_split_decode_ranges() {
    graph_builder   b = make_split_graph();
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "split graph plans");
    check(p.ranges.size() == 3, "three ranges, got " + std::to_string(p.ranges.size()));
    check(p.ranges[0].device == 0 && !p.ranges[0].executor, "range 0 runs on the original device");
    check(p.ranges[1].device == 1 && p.ranges[1].executor, "range 1 is the device-1 executor range");
    check(p.ranges[2].device == 0 && !p.ranges[2].executor, "range 2 is the output head on device 0");
    check(p.ranges[1].begin == b.node("norm-2"), "device 1 starts at layer 2's first node");
    check(p.ranges[2].begin == b.node("result_norm"), "device 0 resumes at result_norm");
    check(p.ranges.back().end == static_cast<int>(b.g.nodes.size()), "ranges cover every node");
    for (size_t r = 1; r < p.ranges.size(); ++r) {
        check(p.ranges[r].begin == p.ranges[r - 1].end, "ranges are contiguous");
    }
}

// The final norm's name carries no layer, and its source is layer
// 3's output. It stays with its producer; result_norm follows the output norm
// weight's placement, not the name of anything.
static void test_tail_follows_weight_placement() {
    graph_builder   b = make_split_graph();
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    check(p.node_device[static_cast<size_t>(b.node("norm"))] == 1, "unnamed norm runs with its producer");
    check(p.node_device[static_cast<size_t>(b.node("result_norm"))] == 0, "result_norm runs where output_norm lives");
    check(p.node_device[static_cast<size_t>(b.node("result_output"))] == 0, "logits run where output.weight lives");

    // Output norm and head on device 1 as well: then the logits would be
    // produced on device 1, and a graph output must stay where llama reads it.
    split_options o{};
    o.output_norm_mask = DEV0 | DEV1;
    o.output_mask      = DEV1;
    graph_builder   b2 = make_split_graph(o);
    dense_exec_plan p2;
    check(dense_exec_build_plan(b2.g, p2) == DENSE_EXEC_GATE_OUTPUT, "a device-1 graph output is refused");
    check(p2.failing_node == b2.node("result_output"), "the refusal names the output node");
}

// Only what crosses a range edge moves: layer 1's output and the control
// leaves go in, the final norm comes back. Weights and the KV cache never move.
static void test_boundary_io() {
    graph_builder   b = make_split_graph();
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    const dense_exec_range_io & io = p.io[1];

    std::vector<int> staged_roots;
    for (int s : io.stage_in) {
        staged_roots.push_back(p.slices[static_cast<size_t>(s)].root);
        check(p.slices[static_cast<size_t>(s)].device == 1, "stage-in lands on the executor device");
    }
    check(staged_roots.size() == 4, "exactly 4 stage-ins, got " + std::to_string(staged_roots.size()));
    check(contains(staged_roots, b.root_index("l_out-1")), "layer 1's output crosses");
    check(contains(staged_roots, b.root_index("inp_pos")), "positions cross");
    check(contains(staged_roots, b.root_index("kq_mask")), "the mask crosses");
    check(contains(staged_roots, b.root_index("kv_idxs")), "KV row ids cross");
    check(!contains(staged_roots, b.root_index("cache_k-2")), "the KV cache is never staged");
    check(!contains(staged_roots, b.root_index("attn_q.weight-2")), "weights are never staged");

    // Everything device 1 produces or stages is published there for the range.
    for (int s : io.stage_in) {
        check(contains(io.publish, s), "staged slices are published");
    }
    check(contains(io.publish, slice_for(p, b.root_index("ffn_inp-3"), 1)), "range-produced tensors are published");
    check(slice_for(p, b.root_index("cache_k-3"), 1) < 0, "no slice is ever made for the KV cache");

    check(io.copy_out.size() == 1, "one copy back, got " + std::to_string(io.copy_out.size()));
    const dense_exec_slice & from = p.slices[static_cast<size_t>(io.copy_out[0].from_slice)];
    const dense_exec_slice & to   = p.slices[static_cast<size_t>(io.copy_out[0].to_slice)];
    check(from.root == b.root_index("norm") && to.root == from.root, "the final norm comes back");
    check(from.device == 1 && to.device == 0, "copied from device 1 to device 0");

    check(p.io[0].stage_in.empty() && p.io[0].publish.empty() && p.io[0].copy_out.empty(),
          "the original-device range is untouched");
    check(p.io[2].stage_in.empty() && p.io[2].publish.empty(), "the head reads through the copy-back only");
}

static void test_noop_keeps_predecessor_device() {
    graph_builder   b = make_split_graph();
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    for (size_t i = 0; i < b.g.nodes.size(); ++i) {
        if (b.g.nodes[i].is_noop) {
            check(i > 0 && p.node_device[i] == p.node_device[i - 1], "no-op " + b.node_names[i] + " never splits");
        }
    }
}

// SET_ROWS writes the KV cache where placement put it. When layer
// 2's KV is not on device 1, the write goes to device 0 and splits layer 2 in
// pieces -- too many ranges. With the range limit lifted, the attention that
// reads that KV on device 1 is refused rather than staged.
static void test_kv_not_on_executor_device() {
    split_options o{};
    o.kv_mask_for_dev1 = DEV0;
    graph_builder   b  = make_split_graph(o);
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_TOO_MANY_RANGES, "interleaved KV writes split ranges");
    check(p.node_device[static_cast<size_t>(b.node("k_set_rows-2"))] == 0, "the KV write follows the KV");

    b.g.max_ranges = 100;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_OPERAND, "device-1 attention cannot read device-0 KV");
    check(p.failing_node == b.node("fattn-2"), "the refusal names the attention node");
}

// A block whose KV was partly re-placed to the host tier (llama.cpp-17ea)
// reads as mixed (-2), and the precheck declines it like host KV.
static void test_mixed_kv_block_is_declined() {
    // Layers 21..29 of a device-0 block demoted; 30..31 on device 1.
    std::vector<int>    owners(32, 0);
    std::vector<size_t> bytes(32, 8u << 20);
    for (int l = 21; l <= 29; ++l) {
        owners[static_cast<size_t>(l)] = -1;
    }
    owners[30] = owners[31] = 1;
    const int kv0           = ggml_sycl::layer_block_kv_device(owners, bytes, 0, 29);
    const int kv1           = ggml_sycl::layer_block_kv_device(owners, bytes, 30, 31);
    check(kv0 == -2, "a partly demoted block's KV owner is mixed");
    check(kv1 == 1, "an untouched block keeps its KV owner");

    dense_exec_precheck_inputs in{};
    in.enabled   = true;
    in.has_graph = true;
    in.has_plan  = true;
    in.blocks.push_back(dense_exec_block{ 0, 29, 0, kv0, false });
    in.blocks.push_back(dense_exec_block{ 30, 31, 1, kv1, false });
    check(dense_exec_first_failing_precheck(in) == DENSE_EXEC_GATE_KV_DEVICE, "mixed KV is refused");
}

// The builder answers from the graph's own KV operands, not the published
// plan: blocks that still claim device KV (a stale or other context's plan)
// do not route a host-resident cache to a device kernel.
static void test_host_kv_operand_under_device_plan() {
    split_options o{};
    o.kv_mask_for_dev1 = 0;  // cache_k-2/-3 live in host memory
    graph_builder b    = make_split_graph(o);
    b.g.max_ranges     = 100;
    dense_exec_plan p;
    check(b.g.blocks[1].kv_device == 1, "the plan still claims device-1 KV");
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_OPERAND, "device-1 attention cannot read host KV");
    check(p.failing_node == b.node("fattn-2"), "the refusal names the attention node");
}

// Placement decides the executor: a host-resident weight is never read over
// PCIe by a GPU, so its node runs on the original device's per-op path.
static void test_host_weight_runs_on_original_device() {
    split_options o{};
    o.host_wq_on_layer2 = true;
    graph_builder   b   = make_split_graph(o);
    dense_exec_plan p;
    (void) dense_exec_build_plan(b.g, p);
    check(p.node_device[static_cast<size_t>(b.node("Qcur-2"))] == 0, "a host weight's matmul stays on device 0");
}

static void test_in_place_write_into_foreign_root() {
    graph_builder b    = make_split_graph();
    // A device-1 node writing into layer 1's output, which device 0 produced.
    const int     l1   = b.root_index("l_out-1");
    const int     norm = b.node("attn_norm-2");
    b.g.nodes.insert(b.g.nodes.begin() + norm + 1, dense_exec_node{ 2, false, l1, { l1 } });
    b.node_names.insert(b.node_names.begin() + norm + 1, "inplace-2");
    for (dense_exec_root & r : b.g.roots) {
        if (r.producer > norm) {
            r.producer++;
        }
    }
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_IN_PLACE, "writing another range's tensor is refused");
    check(p.failing_node == norm + 1, "the refusal names the writer");
}

static void test_arena_layout() {
    graph_builder b = make_split_graph();
    // Odd sizes so alignment is visible.
    for (dense_exec_root & r : b.g.roots) {
        r.bytes = 1000;
    }
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    size_t expect[2] = { 0, 0 };
    for (const dense_exec_slice & s : p.slices) {
        check(s.offset % dense_exec_slice_alignment == 0, "slices are 256-byte aligned");
        check(s.offset == expect[s.device], "slices are packed in creation order");
        expect[s.device] += 1024;
    }
    check(p.arena_bytes[0] == expect[0] && p.arena_bytes[1] == expect[1], "arena totals match the slices");
    check(p.arena_bytes[0] == 1024, "device 0 holds only the copied-back norm");

    b.g.max_arena_bytes = p.arena_bytes[1] - 1;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_ARENA_TOO_LARGE, "arena cap is enforced");
}

static void test_single_device_graph() {
    graph_builder b                = make_split_graph();
    b.g.blocks[1].execution_device = 0;
    for (dense_exec_root & r : b.g.roots) {
        if (r.kind != DENSE_EXEC_ROOT_NODE && r.kind != DENSE_EXEC_ROOT_CONTROL) {
            r.resident_mask = DEV0;
        }
    }
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_SINGLE_DEVICE, "nothing to execute elsewhere");
}

// Two executor ranges on the same device: a tensor the first produced is
// published again for the second, not copied through device 0.
static void test_same_device_reuse_across_ranges() {
    graph_builder b;
    b.g.original_device = 0;
    b.g.blocks          = {
        { 0, 0, 0, 0, false },
        { 1, 1, 1, 1, false },
        { 2, 2, 0, 0, false },
        { 3, 3, 1, 1, false },
    };
    const int w0 = b.root("w0", DENSE_EXEC_ROOT_WEIGHT, DEV0);
    const int w1 = b.root("w1", DENSE_EXEC_ROOT_WEIGHT, DEV1);
    const int in = b.root("inp", DENSE_EXEC_ROOT_CONTROL);
    const int a0 = b.op("a-0", 0, { in, w0 });
    const int a1 = b.op("a-1", 1, { a0, w1 });
    const int a2 = b.op("a-2", 2, { a1, w0 });
    const int a3 = b.op("a-3", 3, { a2, a1, w1 });
    (void) a3;
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    check(dense_exec_plan_violation(b.g, p) == nullptr, "a four-range plan is clean");
    check(p.ranges.size() == 4, "four ranges");
    const int s = slice_for(p, a1, 1);
    check(s >= 0 && contains(p.io[3].publish, s), "a-1 is republished on device 1");
    check(!contains(p.io[3].stage_in, s), "a-1 is not re-staged");
    check(contains(p.io[3].stage_in, slice_for(p, a2, 1)), "a-2 from device 0 is staged");
    check(p.io[1].copy_out.size() == 1 && p.slices[static_cast<size_t>(p.io[1].copy_out[0].to_slice)].root == a1,
          "a-1 is copied back once for device 0's a-2");
    check(contains(p.io[1].stage_in, slice_for(p, in, 1)) == false, "a control leaf only reaches consumers' ranges");
}

static void expect_violation(const dense_exec_graph & g, const dense_exec_plan & p, const char * expected) {
    const char * got = dense_exec_plan_violation(g, p);
    check(got != nullptr && std::strcmp(got, expected) == 0,
          std::string("expected violation '") + expected + "', got '" + (got ? got : "none") + "'");
}

// Every plan the planner builds keeps the publication invariants, and the
// check is not vacuous: each corruption below is a way a plan could drop or
// alias a live owner, and each is caught.
static void test_plan_invariants() {
    graph_builder   b = make_split_graph();
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    const char * clean = dense_exec_plan_violation(b.g, p);
    check(clean == nullptr, std::string("a built plan is clean, got ") + (clean ? clean : ""));

    {
        dense_exec_plan  bad  = p;
        dense_exec_slice copy = bad.slices[static_cast<size_t>(bad.io[1].publish[0])];
        bad.slices.push_back(copy);
        expect_violation(b.g, bad, "two slices for one root on one device");
    }
    {
        // A source override on the root of a result the range publishes.
        dense_exec_plan bad = p;
        bad.io[1].stage_in.push_back(slice_for(bad, b.root_index("ffn_inp-3"), 1));
        expect_violation(b.g, bad, "a range stages over a root it produces");
    }
    {
        dense_exec_plan bad = p;
        bad.io[1].publish.push_back(bad.io[1].publish[0]);
        expect_violation(b.g, bad, "a range publishes one root twice");
    }
    {
        dense_exec_plan  bad = p;
        dense_exec_slice foreign{};
        foreign.root   = b.root_index("attn_q.weight-2");
        foreign.device = 0;
        bad.slices.push_back(foreign);
        bad.io[1].publish.push_back(static_cast<int>(bad.slices.size()) - 1);
        expect_violation(b.g, bad, "a range publishes a slice of another device");
    }
    {
        dense_exec_plan bad = p;
        std::swap(bad.io[1].copy_out[0].from_slice, bad.io[1].copy_out[0].to_slice);
        expect_violation(b.g, bad, "a copy out does not go from the range's device to the original device");
    }
    {
        dense_exec_plan bad = p;
        bad.io[1].stage_in.push_back(bad.io[1].copy_out[0].to_slice);
        expect_violation(b.g, bad, "a staged slice is not published");
    }
    {
        dense_exec_plan bad = p;
        bad.io[0].publish.push_back(bad.io[1].publish[0]);
        expect_violation(b.g, bad, "an original-device range carries io");
    }
}

// Each direction of a crossing moves through one host staging buffer per
// device: its copies pack into disjoint aligned spans, so a range's copies
// can be submitted together and waited on once.
static void test_host_stage_layout() {
    graph_builder b = make_split_graph();
    for (dense_exec_root & r : b.g.roots) {
        r.bytes = 1000;
    }
    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    check(dense_exec_plan_violation(b.g, p) == nullptr, "a built plan is clean");
    check(!p.io[1].stage_in.empty() && !p.io[1].copy_out.empty(), "range 1 crosses both ways");

    size_t widest = 0;
    for (const dense_exec_range_io & io : p.io) {
        check(io.stage_in_host.size() == io.stage_in.size(), "one host offset per staged slice");
        check(io.copy_out_host.size() == io.copy_out.size(), "one host offset per copy out");
        size_t expect = 0;
        for (size_t k = 0; k < io.stage_in.size(); ++k) {
            check(io.stage_in_host[k] == expect, "staged slices pack in order");
            check(io.stage_in_host[k] % dense_exec_slice_alignment == 0, "host spans are aligned");
            expect += dense_exec_align(p.slices[static_cast<size_t>(io.stage_in[k])].bytes);
        }
        widest = std::max(widest, expect);
        expect = 0;
        for (size_t k = 0; k < io.copy_out.size(); ++k) {
            check(io.copy_out_host[k] == expect, "copies out pack in order");
            expect += dense_exec_align(p.slices[static_cast<size_t>(io.copy_out[k].from_slice)].bytes);
        }
        widest = std::max(widest, expect);
    }
    check(p.host_stage_bytes == widest, "the host buffer holds the widest direction of any range");
    check(p.io[1].stage_in.size() > 1 && p.host_stage_bytes == 1024 * p.io[1].stage_in.size(),
          "1000-byte roots take 1024 host bytes each");

    {
        dense_exec_plan bad            = p;
        bad.io[1].stage_in_host.back() = bad.io[1].stage_in_host.front();
        expect_violation(b.g, bad, "host staging spans overlap or leave the buffer");
    }
    {
        dense_exec_plan bad  = p;
        bad.host_stage_bytes = bad.io[1].stage_in_host.back() + 999;  // one byte short of the last span
        expect_violation(b.g, bad, "host staging spans overlap or leave the buffer");
    }
    {
        dense_exec_plan bad = p;
        bad.io[1].copy_out_host.clear();
        expect_violation(b.g, bad, "host staging spans overlap or leave the buffer");
    }
}

// A range graph takes the pool scratch freed while it recorded: only what the
// pool gained since the recording began, in order, and nothing from before.
static void test_take_since() {
    std::vector<int> pool  = { 1, 2, 3, 4, 5 };
    std::vector<int> graph = { 9 };
    check(dense_exec_take_since(pool, 2, graph) == 3, "moves the three entries after the baseline");
    check(pool == std::vector<int>({ 1, 2 }), "the pool keeps what it held before the baseline");
    check(graph == std::vector<int>({ 9, 3, 4, 5 }), "the graph appends the moved entries in order");

    check(dense_exec_take_since(pool, 2, graph) == 0, "nothing new since the baseline moves nothing");
    check(pool.size() == 2 && graph.size() == 4, "an empty move changes neither side");

    std::vector<int> drained = { 7 };
    std::vector<int> other;
    check(dense_exec_take_since(drained, 3, other) == 1, "a pool drained since the baseline gives up all it holds");
    check(drained.empty() && other == std::vector<int>({ 7 }), "and it ends empty");
}

// dev0 | dev1 | dev0 | dev1. Range 1 produces r-1; a device-0 node in range 2
// writes r-1 in place; range 3 reads r-1 on device 1. Range 3 could only
// re-publish range 1's device-1 slice -- the value from before the write --
// so the in-place write on the original device must be refused.
static void test_original_range_writes_executor_result() {
    graph_builder b;
    b.g.original_device = 0;
    b.g.blocks          = {
        { 0, 0, 0, 0, false },
        { 1, 1, 1, 1, false },
        { 2, 2, 0, 0, false },
        { 3, 3, 1, 1, false },
    };
    const int w0 = b.root("w0", DENSE_EXEC_ROOT_WEIGHT, DEV0);
    const int w1 = b.root("w1", DENSE_EXEC_ROOT_WEIGHT, DEV1);
    const int in = b.root("inp", DENSE_EXEC_ROOT_CONTROL, DEV0);
    const int a0 = b.op("a-0", 0, { in, w0 });
    const int r1 = b.op("r-1", 1, { a0, w1 });
    b.write_into("inplace-2", 2, r1, { r1, w0 });
    const int a3 = b.op("a-3", 3, { r1, w1 });
    (void) a3;

    dense_exec_plan       p;
    const dense_exec_gate gate = dense_exec_build_plan(b.g, p);
    check(
        gate == DENSE_EXEC_GATE_IN_PLACE,
        std::string("an original-device write into an executor result is refused, got ") + dense_exec_gate_name(gate));
    check(p.failing_node == b.node("inplace-2"), "the refusal names the writer");
}

// The same hazard seen by the invariant check: a plan built before the
// in-place write existed does not cover it.
static void test_violation_original_range_writes_executor_result() {
    graph_builder b;
    b.g.original_device = 0;
    b.g.blocks          = {
        { 0, 0, 0, 0, false },
        { 1, 1, 1, 1, false },
        { 2, 2, 0, 0, false },
        { 3, 3, 1, 1, false },
    };
    const int w0 = b.root("w0", DENSE_EXEC_ROOT_WEIGHT, DEV0);
    const int w1 = b.root("w1", DENSE_EXEC_ROOT_WEIGHT, DEV1);
    const int in = b.root("inp", DENSE_EXEC_ROOT_CONTROL, DEV0);
    const int a0 = b.op("a-0", 0, { in, w0 });
    const int r1 = b.op("r-1", 1, { a0, w1 });
    const int a2 = b.op("a-2", 2, { r1, w0 });
    b.op("a-3", 3, { r1, a2, w1 });

    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    check(dense_exec_plan_violation(b.g, p) == nullptr, "clean before the write");
    b.g.nodes[static_cast<size_t>(b.node("a-2"))].dst_root = r1;
    expect_violation(b.g, p, "an original-device node writes an executor range's result");
}

// A control leaf the executor device can already read in place is published
// by nobody and copied by nobody.
static void test_control_resident_on_executor_is_not_staged() {
    graph_builder b;
    b.g.original_device = 0;
    b.g.blocks          = {
        { 0, 0, 0, 0, false },
        { 1, 1, 1, 1, false },
    };
    const int w0   = b.root("w0", DENSE_EXEC_ROOT_WEIGHT, DEV0);
    const int w1   = b.root("w1", DENSE_EXEC_ROOT_WEIGHT, DEV1);
    const int both = b.root("ctl-both", DENSE_EXEC_ROOT_CONTROL, DEV0 | DEV1);
    const int dev0 = b.root("ctl-dev0", DENSE_EXEC_ROOT_CONTROL, DEV0);
    const int a0   = b.op("a-0", 0, { both, w0 });
    b.op("a-1", 1, { a0, both, dev0, w1 });
    b.op("out", 0, { b.root_index("a-1"), w0 });

    dense_exec_plan p;
    check(dense_exec_build_plan(b.g, p) == DENSE_EXEC_GATE_NONE, "plans");
    check(p.ranges.size() == 3 && p.ranges[1].executor, "one executor range");
    check(slice_for(p, both, 1) < 0, "a control leaf resident on device 1 gets no slice");
    check(contains(p.io[1].stage_in, slice_for(p, dev0, 1)), "a device-0-only control leaf is staged");
    check(p.io[1].stage_in.size() == 2, "a-0 and ctl-dev0 are the only stage-ins");
}

int main() {
    struct test_case {
        const char * name;
        void (*fn)();
    };

    const test_case cases[] = {
        { "precheck-gates",                             test_precheck_gates                                  },
        { "split-decode-ranges",                        test_split_decode_ranges                             },
        { "tail-follows-weight-placement",              test_tail_follows_weight_placement                   },
        { "boundary-io",                                test_boundary_io                                     },
        { "noop-keeps-predecessor-device",              test_noop_keeps_predecessor_device                   },
        { "kv-not-on-executor-device",                  test_kv_not_on_executor_device                       },
        { "mixed-kv-block-is-declined",                 test_mixed_kv_block_is_declined                      },
        { "host-kv-operand-under-device-plan",          test_host_kv_operand_under_device_plan               },
        { "host-weight-runs-on-original-device",        test_host_weight_runs_on_original_device             },
        { "in-place-write-into-foreign-root",           test_in_place_write_into_foreign_root                },
        { "arena-layout",                               test_arena_layout                                    },
        { "single-device-graph",                        test_single_device_graph                             },
        { "same-device-reuse-across-ranges",            test_same_device_reuse_across_ranges                 },
        { "plan-invariants",                            test_plan_invariants                                 },
        { "original-range-writes-executor-result",      test_original_range_writes_executor_result           },
        { "violation-original-writes-executor-result",  test_violation_original_range_writes_executor_result },
        { "control-resident-on-executor-is-not-staged", test_control_resident_on_executor_is_not_staged      },
        { "graph-off-reasons",                          test_graph_off_reasons                               },
        { "graph-off-debug-envs",                       test_graph_off_debug_envs                            },
        { "host-stage-layout",                          test_host_stage_layout                               },
        { "take-since",                                 test_take_since                                      },
    };

    int failed = 0;
    for (const test_case & tc : cases) {
        try {
            tc.fn();
            std::cout << "PASS " << tc.name << "\n";
        } catch (const std::exception & e) {
            std::cout << "FAIL " << tc.name << ": " << e.what() << "\n";
            failed++;
        }
    }
    std::cout << (failed == 0 ? "ALL PASS" : "FAILED") << " (" << failed << " failed)\n";
    return failed == 0 ? 0 : 1;
}
