// Canary D0.3 — single-device op_id stability across repeated graph_reserve.
//
// Question (from docs/plans/2026-04-24-planner-validation-phases.md §Task 1):
// can plan.ops (C2's planner data structure) safely key on op_id, where
// op_id is the graph node's index in the order ggml emits ask=true callbacks?
// Equivalently: does the SYCL/CPU scheduler produce identical (op_id, op_type,
// name) sequences when graph_reserve runs five times against the same context
// with the same inputs?
//
// Original D0.3 was scoped as a multi-device CPY-visibility check; that
// scenario remains unavailable on this host (single SYCL device, B50
// PCI-disabled per host policy; see CLAUDE.md memory feedback_disable_b50.md).
// The op_id-stability question is the load-bearing claim for C2 and only
// needs single-context determinism, so this revision answers that question
// directly. Multi-device CPY-name stability is preserved as a future TODO.
//
// Execution model:
//   * CPU backend via ONEAPI_DEVICE_SELECTOR=opencl:cpu so we sidestep the
//     m09zb L0 DirectSubmission wedge that gates GPU-backend canaries — this
//     question is about ggml scheduler determinism, not SYCL-specific
//     behavior.
//   * One process loads Mistral 7B once, creates one context, runs 5
//     llama_decode() calls with identical 1-token batches, collecting per-run
//     (op_id, op_type, name) sequences via cb_eval. PASS iff all 5 captures
//     are identical; FAIL with first divergence index otherwise.

#include "common.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "llama.h"
#include "test-planner-canary-common.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace planner_canary;

static const char * MD_PATH        = "docs/plans/data/planner-canaries/d0.3-cpy-visibility.md";
static const char * JSON_PATH      = "tests/data/planner-canaries/d0.3.json";
static const char * MD_PATH_D03A   = "docs/plans/data/planner-canaries/d0.3a-multidevice-cpy-visibility.md";
static const char * JSON_PATH_D03A = "tests/data/planner-canaries/d0.3a.json";

// Per-node capture during one decode call. op_id is the position in the
// ask-pass callback sequence — the same indexing C2's plan.ops would use
// if it keyed on graph node index.
struct node_triple {
    int         op_id;
    int         op_type;  // ggml_op enum value
    std::string name;     // ggml_tensor::name (may be empty)

    bool operator==(const node_triple & o) const { return op_id == o.op_id && op_type == o.op_type && name == o.name; }

    bool operator!=(const node_triple & o) const { return !(*this == o); }
};

// Worker-global capture buffer for the cb_eval callback. The callback is C
// ABI and ggml passes user_data through it, but we keep the buffer here as
// a single-threaded scratch since each decode runs serially.
static std::vector<node_triple> g_current;

struct cpy_snapshot {
    int                      n_backends             = 0;
    int                      n_splits               = 0;
    int                      n_copies               = 0;
    int                      n_nodes                = 0;
    int                      n_leafs                = 0;
    int                      n_cpy                  = 0;
    int                      n_scheduler_copy_edges = 0;
    std::vector<std::string> cpy_names;
    std::vector<std::string> scheduler_copy_names;
    std::vector<std::string> backend_names;
};

static bool op_snapshot_cb(struct ggml_tensor * t, bool ask, void * /*user*/) {
    if (!ask) {
        return true;  // proceed with compute on the eval-pass
    }
    // Skip leaf nodes (op == GGML_OP_NONE) and null tensors — they're stable too
    // but the question is "does the OP-id sequence match" and leaves are noise.
    if (t == nullptr || t->op == GGML_OP_NONE) {
        return false;
    }
    // ggml_tensor::name is a fixed-size char array, always present;
    // empty when ggml has not assigned a debug name to the node.
    g_current.push_back({ (int) g_current.size(), (int) t->op, std::string(t->name) });
    return false;  // don't force per-node sync
}

// Render a triple as "[op_id]op_name:tensor_name" for diagnostic output.
static std::string fmt_triple(const node_triple & t) {
    std::ostringstream oss;
    oss << "[" << t.op_id << "]" << ggml_op_name((enum ggml_op) t.op_type) << ":" << t.name;
    return oss.str();
}

static cpy_snapshot snapshot_split_graph(ggml_backend_sched_t sched) {
    cpy_snapshot s;
    s.n_backends = ggml_backend_sched_get_n_backends(sched);
    s.n_splits   = ggml_backend_sched_get_n_splits(sched);
    s.n_copies   = ggml_backend_sched_get_n_copies(sched);

    for (int i = 0; i < s.n_backends; ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        s.backend_names.push_back(backend ? ggml_backend_name(backend) : "null");
    }

    const ggml_cgraph * graph = ggml_backend_sched_get_debug_graph(sched);
    if (graph == nullptr) {
        return s;
    }

    s.n_nodes = graph->n_nodes;
    s.n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node == nullptr) {
            continue;
        }
        const std::string name = node->name;
        if (node->op == GGML_OP_CPY) {
            ++s.n_cpy;
            s.cpy_names.push_back(name);
        }
        if (name.find("#d0_3a_") != std::string::npos) {
            ++s.n_scheduler_copy_edges;
            s.scheduler_copy_names.push_back(name);
        }
    }
    for (int i = 0; i < graph->n_leafs; ++i) {
        const ggml_tensor * leaf = graph->leafs[i];
        if (leaf == nullptr) {
            continue;
        }
        const std::string name = leaf->name;
        if (name.find("#d0_3a_") != std::string::npos) {
            ++s.n_scheduler_copy_edges;
            s.scheduler_copy_names.push_back(name);
        }
    }
    return s;
}

static bool is_gpu_dev(ggml_backend_dev_t dev) {
    const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
    return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

static bool is_cpu_dev(ggml_backend_dev_t dev) {
    return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

static std::string join_strings(const std::vector<std::string> & values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << values[i];
    }
    return oss.str();
}

static int run_multidevice_cpy_visibility() {
    findings f;
    f.canary_id = "D0.3a";
    f.result    = status::FAIL;

    const char * split_ratio = std::getenv("GGML_SYCL_SPLIT_RATIO");
    add(f, "mode", "synthetic scheduler multi-device CPY visibility");
    add(f, "GGML_SYCL_SPLIT_RATIO", split_ratio ? split_ratio : "(unset)");

    if (split_ratio == nullptr || split_ratio[0] == '\0') {
        f.result  = status::INCONCLUSIVE;
        f.summary = "GGML_SYCL_SPLIT_RATIO is not set, so the scheduler will not expose multiple SYCL devices";
        f.recommendation =
            "Rerun with D0_3_MULTIDEVICE=1 GGML_SYCL_SPLIT_RATIO=50,50 and SYCL_DEVICE_FILTER=level_zero:gpu";
        write_markdown(f, MD_PATH_D03A);
        write_json(f, JSON_PATH_D03A);
        return 0;
    }

    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends;
    std::vector<std::string>    backend_names;
    for (size_t i = 0; i < ggml_backend_dev_count() && backends.size() < 2; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !is_gpu_dev(dev)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        backend_names.emplace_back(ggml_backend_dev_name(dev));
        backends.push_back(backend);
    }

    if (backends.size() < 2) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "Fewer than two SYCL GPU backends were available";
        f.recommendation = "Rerun on a host where GGML_SYCL_SPLIT_RATIO exposes at least two GPU devices";
        add(f, "gpu_backends_found", std::to_string(backends.size()));
        add(f, "backend_names", join_strings(backend_names));
        write_markdown(f, MD_PATH_D03A);
        write_json(f, JSON_PATH_D03A);
        for (ggml_backend_t backend : backends) {
            ggml_backend_free(backend);
        }
        return 0;
    }

    bool cpu_backend_found = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !is_cpu_dev(dev)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        backend_names.emplace_back(ggml_backend_dev_name(dev));
        backends.push_back(backend);
        cpu_backend_found = true;
        break;
    }

    if (!cpu_backend_found) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "CPU backend was unavailable; ggml_backend_sched_new requires CPU as the final backend";
        f.recommendation = "Ensure ggml_backend_load_all registers the CPU backend before rerunning D0.3a";
        add(f, "backend_names", join_strings(backend_names));
        write_markdown(f, MD_PATH_D03A);
        write_json(f, JSON_PATH_D03A);
        for (ggml_backend_t backend : backends) {
            ggml_backend_free(backend);
        }
        return 0;
    }

    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) {
        f.summary        = "ggml_backend_sched_new failed";
        f.recommendation = "Investigate scheduler construction for multi-device SYCL backends";
        write_markdown(f, MD_PATH_D03A);
        write_json(f, JSON_PATH_D03A);
        for (ggml_backend_t backend : backends) {
            ggml_backend_free(backend);
        }
        return 1;
    }

    constexpr int             N_RUNS = 2;
    std::vector<cpy_snapshot> runs;
    runs.reserve(N_RUNS);

    for (int r = 0; r < N_RUNS; ++r) {
        ggml_init_params params = {
            /*.mem_size   =*/1024 * 1024,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_set_name(a, "d0_3a_input_a");
        ggml_set_name(b, "d0_3a_input_b");
        ggml_set_input(a);
        ggml_set_input(b);

        ggml_tensor * sum = ggml_add(ctx, a, b);
        ggml_set_name(sum, "d0_3a_sum_on_backend1");
        ggml_set_output(sum);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sum);

        ggml_backend_sched_reset(sched);
        ggml_backend_sched_set_tensor_backend(sched, a, backends[0]);
        ggml_backend_sched_set_tensor_backend(sched, b, backends[0]);
        ggml_backend_sched_set_tensor_backend(sched, sum, backends[1]);
        ggml_backend_sched_split_graph(sched, graph);

        runs.push_back(snapshot_split_graph(sched));
        fprintf(stderr,
                "[D0.3a] synthetic run %d: backends=%d splits=%d copies=%d nodes=%d leafs=%d cpy=%d "
                "scheduler_copy_edges=%d\n",
                r + 1, runs.back().n_backends, runs.back().n_splits, runs.back().n_copies, runs.back().n_nodes,
                runs.back().n_leafs, runs.back().n_cpy, runs.back().n_scheduler_copy_edges);
        ggml_free(ctx);
    }

    const bool has_multiple_backends = runs[0].n_backends >= 3;
    const bool has_cpy_nodes         = runs[0].n_cpy > 0;
    const bool has_copy_edges        = runs[0].n_scheduler_copy_edges > 0;
    const bool stable_names          = runs.size() == N_RUNS && runs[0].cpy_names == runs[1].cpy_names;
    const bool stable_copy_edges =
        runs.size() == N_RUNS && runs[0].scheduler_copy_names == runs[1].scheduler_copy_names;

    f.result         = (has_multiple_backends && has_cpy_nodes && stable_names) ? status::PASS : status::FAIL;
    f.summary        = f.result == status::PASS ?
                           "Multi-device split graph exposes stable GGML_OP_CPY nodes across repeated contexts" :
                           "Scheduler exposes stable copy-edge tensors, but not as GGML_OP_CPY graph nodes";
    f.recommendation = f.result == status::PASS ?
                           "C2 can include post-split CPY nodes in plan.ops keying for this split configuration" :
                           "Revise C2 to read scheduler split-input copy edges instead of assuming cross-backend "
                           "transfers appear as GGML_OP_CPY nodes";

    add(f, "backend_names", join_strings(backend_names));
    add(f, "synthetic_graph", "add(input_a@backend0,input_b@backend0)->sum@backend1");
    add(f, "runs", std::to_string(runs.size()));
    add(f, "run0_backends", std::to_string(runs[0].n_backends));
    add(f, "run0_backend_names", join_strings(runs[0].backend_names));
    add(f, "run0_splits", std::to_string(runs[0].n_splits));
    add(f, "run0_copies", std::to_string(runs[0].n_copies));
    add(f, "run0_nodes", std::to_string(runs[0].n_nodes));
    add(f, "run0_leafs", std::to_string(runs[0].n_leafs));
    add(f, "run0_cpy_nodes", std::to_string(runs[0].n_cpy));
    add(f, "run0_scheduler_copy_edges", std::to_string(runs[0].n_scheduler_copy_edges));
    add(f, "has_scheduler_copy_edges", has_copy_edges ? "true" : "false");
    if (runs.size() > 1) {
        add(f, "run1_cpy_nodes", std::to_string(runs[1].n_cpy));
        add(f, "run1_scheduler_copy_edges", std::to_string(runs[1].n_scheduler_copy_edges));
        add(f, "cpy_names_stable", stable_names ? "true" : "false");
        add(f, "scheduler_copy_edge_names_stable", stable_copy_edges ? "true" : "false");
    }
    if (!runs[0].cpy_names.empty()) {
        add(f, "first_cpy_name", runs[0].cpy_names.front());
        add(f, "last_cpy_name", runs[0].cpy_names.back());
    }
    if (!runs[0].scheduler_copy_names.empty()) {
        add(f, "first_scheduler_copy_edge", runs[0].scheduler_copy_names.front());
        add(f, "last_scheduler_copy_edge", runs[0].scheduler_copy_names.back());
    }
    add(f, "plan_section", "docs/plans/2026-04-22-unified-memory-placement-plan.md D0.3");

    write_markdown(f, MD_PATH_D03A);
    write_json(f, JSON_PATH_D03A);

    ggml_backend_sched_free(sched);
    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    return f.result == status::PASS ? 0 : 1;
}

int main(int /*argc*/, char ** /*argv*/) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    if (std::getenv("D0_3_MULTIDEVICE") != nullptr) {
        return run_multidevice_cpy_visibility();
    }

    findings f;
    f.canary_id = "D0.3";
    f.result    = status::FAIL;

    llama_backend_init();

    const std::string mp = mistral_path();
    if (access(mp.c_str(), R_OK) != 0) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "Mistral 7B GGUF unavailable at expected path";
        f.recommendation = "Set MISTRAL_PATH to a readable GGUF and re-run";
        add(f, "model_path", mp);
        write_markdown(f, MD_PATH);
        write_json(f, JSON_PATH);
        llama_backend_free();
        return 0;
    }

    // CPU-only: n_gpu_layers=0 keeps weights on host and the CPU backend
    // executes every node, sidestepping m09zb. The ask-pass callback fires
    // identically regardless of which backend ultimately computes a node, so
    // op-id ordering is observable here.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;

    fprintf(stderr, "[D0.3] loading model: %s\n", mp.c_str());
    llama_model * model = llama_model_load_from_file(mp.c_str(), mparams);
    if (model == nullptr) {
        f.summary        = "llama_model_load_from_file failed";
        f.recommendation = "Investigate model load failure; canary cannot proceed";
        add(f, "model_path", mp);
        write_markdown(f, MD_PATH);
        write_json(f, JSON_PATH);
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 1024;
    cparams.n_batch              = 512;
    cparams.n_ubatch             = 512;
    cparams.cb_eval              = op_snapshot_cb;
    cparams.cb_eval_user_data    = nullptr;

    fprintf(stderr, "[D0.3] creating context (n_ctx=%u, ubatch=%u)\n", cparams.n_ctx, cparams.n_ubatch);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        f.summary        = "llama_init_from_model failed";
        f.recommendation = "Investigate context creation failure; canary cannot proceed";
        write_markdown(f, MD_PATH);
        write_json(f, JSON_PATH);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    constexpr int                         N_RUNS = 5;
    std::vector<std::vector<node_triple>> runs;
    runs.reserve(N_RUNS);

    const llama_token bos           = llama_vocab_bos(llama_model_get_vocab(model));
    bool              decode_failed = false;
    int               decode_rc     = 0;

    llama_memory_t mem = llama_get_memory(ctx);

    for (int r = 0; r < N_RUNS; ++r) {
        g_current.clear();

        // Clear KV state so every run starts at pos=0 with an empty cache —
        // ensures the graph shape is identical (no token-position-dependent
        // attention masking divergence). data=false avoids zeroing the
        // backing buffers; metadata reset is sufficient to make pos=0 valid
        // for the next decode.
        llama_memory_clear(mem, /*data=*/false);

        // Single-token decode at pos=0 — identical inputs across runs.
        // n_seq_max=1 matches the single-sequence pattern used by D0.2.
        llama_batch batch = llama_batch_init(/*n_tokens=*/1, /*embd=*/0, /*n_seq_max=*/1);
        common_batch_add(batch, bos, /*pos=*/0, /*seq_ids=*/{ 0 }, /*logits=*/true);

        fprintf(stderr, "[D0.3] run %d/%d: llama_decode\n", r + 1, N_RUNS);
        decode_rc = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (decode_rc != 0) {
            // Non-fatal status (e.g. KV-cache eviction warning) is acceptable
            // since the ask-pass populates g_current before compute; abort
            // captures are still indexable. Hard failure (negative rc) is
            // treated as a canary infrastructure failure.
            fprintf(stderr, "[D0.3] run %d: llama_decode rc=%d\n", r + 1, decode_rc);
            if (decode_rc < 0) {
                decode_failed = true;
                break;
            }
        }

        fprintf(stderr, "[D0.3] run %d: captured %zu nodes\n", r + 1, g_current.size());
        runs.push_back(g_current);
    }

    if (decode_failed || runs.size() != N_RUNS) {
        f.result         = status::FAIL;
        f.summary        = "llama_decode hard-failed before all 5 runs completed";
        f.recommendation = "Investigate decode failure; canary cannot answer the op_id-stability question";
        add(f, "runs_completed", std::to_string(runs.size()));
        add(f, "last_decode_rc", std::to_string(decode_rc));
        write_markdown(f, MD_PATH);
        write_json(f, JSON_PATH);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Compare every later run to run 0. PASS iff all match.
    bool        all_same      = true;
    int         diverge_run   = -1;
    int         diverge_index = -1;
    std::string diverge_a, diverge_b;
    for (size_t r = 1; r < runs.size() && all_same; ++r) {
        if (runs[r].size() != runs[0].size()) {
            all_same      = false;
            diverge_run   = (int) r;
            diverge_index = -1;  // size mismatch, no specific index
            break;
        }
        for (size_t i = 0; i < runs[r].size(); ++i) {
            if (runs[r][i] != runs[0][i]) {
                all_same      = false;
                diverge_run   = (int) r;
                diverge_index = (int) i;
                diverge_a     = fmt_triple(runs[0][i]);
                diverge_b     = fmt_triple(runs[r][i]);
                break;
            }
        }
    }

    f.result  = all_same ? status::PASS : status::FAIL;
    f.summary = all_same ? "op_id ordering identical across 5 graph_reserve calls — plan.ops can safely key on op_id" :
                           "op_id ordering diverges across runs — plan.ops cannot safely key on op_id";
    f.recommendation =
        all_same ? "A3a/C2 can proceed with op_id keying as specified" :
                   "Switch plan.ops keying to a more stable identifier (e.g. hash of (op_type, source-tensor names))";

    add(f, "runs", std::to_string(runs.size()));
    add(f, "nodes_per_run", std::to_string(runs[0].size()));
    add(f, "backend", "cpu (ONEAPI_DEVICE_SELECTOR=opencl:cpu sidesteps m09zb)");
    add(f, "model_path", mp);
    add(f, "n_ctx", std::to_string(cparams.n_ctx));
    add(f, "n_ubatch", std::to_string(cparams.n_ubatch));
    if (!all_same) {
        add(f, "first_divergence_run", std::to_string(diverge_run));
        add(f, "first_divergence_index", std::to_string(diverge_index));
        if (diverge_index >= 0) {
            add(f, "first_divergence_run0", diverge_a);
            add(f, "first_divergence_runN", diverge_b);
        } else {
            add(f, "first_divergence_size_run0", std::to_string(runs[0].size()));
            add(f, "first_divergence_size_runN", std::to_string(runs[diverge_run].size()));
        }
    }
    add(f, "todo_multidevice",
        "Multi-device CPY-name stability check is the original D0.3 scope; "
        "preserve as future work once >=2 SYCL devices are usable "
        "on this host (B50 disabled per CLAUDE.md memory feedback_disable_b50.md) "
        "and E1 staging_buffer_pool fix (llama.cpp-m09zb) lands");
    add(f, "plan_section", "docs/plans/2026-04-24-planner-validation-phases.md Task 1");

    write_markdown(f, MD_PATH);
    write_json(f, JSON_PATH);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return all_same ? 0 : 1;
}
