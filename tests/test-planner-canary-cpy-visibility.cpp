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

#include "test-planner-canary-common.hpp"
#include "llama.h"
#include "common.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace planner_canary;

static const char * MD_PATH   = "docs/plans/data/planner-canaries/d0.3-cpy-visibility.md";
static const char * JSON_PATH = "tests/data/planner-canaries/d0.3.json";

// Per-node capture during one decode call. op_id is the position in the
// ask-pass callback sequence — the same indexing C2's plan.ops would use
// if it keyed on graph node index.
struct node_triple {
    int         op_id;
    int         op_type;  // ggml_op enum value
    std::string name;     // ggml_tensor::name (may be empty)

    bool operator==(const node_triple & o) const {
        return op_id == o.op_id && op_type == o.op_type && name == o.name;
    }
    bool operator!=(const node_triple & o) const { return !(*this == o); }
};

// Worker-global capture buffer for the cb_eval callback. The callback is C
// ABI and ggml passes user_data through it, but we keep the buffer here as
// a single-threaded scratch since each decode runs serially.
static std::vector<node_triple> g_current;

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
    g_current.push_back({(int) g_current.size(),
                         (int) t->op,
                         std::string(t->name)});
    return false;  // don't force per-node sync
}

// Render a triple as "[op_id]op_name:tensor_name" for diagnostic output.
static std::string fmt_triple(const node_triple & t) {
    std::ostringstream oss;
    oss << "[" << t.op_id << "]"
        << ggml_op_name((enum ggml_op) t.op_type) << ":" << t.name;
    return oss.str();
}

int main(int /*argc*/, char ** /*argv*/) {
    setvbuf(stdout, nullptr, _IONBF, 0);

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
        write_json    (f, JSON_PATH);
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
        write_json    (f, JSON_PATH);
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 1024;
    cparams.n_batch           = 512;
    cparams.n_ubatch          = 512;
    cparams.cb_eval           = op_snapshot_cb;
    cparams.cb_eval_user_data = nullptr;

    fprintf(stderr, "[D0.3] creating context (n_ctx=%u, ubatch=%u)\n",
            cparams.n_ctx, cparams.n_ubatch);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        f.summary        = "llama_init_from_model failed";
        f.recommendation = "Investigate context creation failure; canary cannot proceed";
        write_markdown(f, MD_PATH);
        write_json    (f, JSON_PATH);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    constexpr int N_RUNS = 5;
    std::vector<std::vector<node_triple>> runs;
    runs.reserve(N_RUNS);

    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));
    bool decode_failed = false;
    int  decode_rc     = 0;

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
        common_batch_add(batch, bos, /*pos=*/0, /*seq_ids=*/{0}, /*logits=*/true);

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

        fprintf(stderr, "[D0.3] run %d: captured %zu nodes\n",
                r + 1, g_current.size());
        runs.push_back(g_current);
    }

    if (decode_failed || runs.size() != N_RUNS) {
        f.result         = status::FAIL;
        f.summary        = "llama_decode hard-failed before all 5 runs completed";
        f.recommendation = "Investigate decode failure; canary cannot answer the op_id-stability question";
        add(f, "runs_completed",  std::to_string(runs.size()));
        add(f, "last_decode_rc",  std::to_string(decode_rc));
        write_markdown(f, MD_PATH);
        write_json    (f, JSON_PATH);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Compare every later run to run 0. PASS iff all match.
    bool all_same = true;
    int  diverge_run    = -1;
    int  diverge_index  = -1;
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

    f.result = all_same ? status::PASS : status::FAIL;
    f.summary = all_same
        ? "op_id ordering identical across 5 graph_reserve calls — plan.ops can safely key on op_id"
        : "op_id ordering diverges across runs — plan.ops cannot safely key on op_id";
    f.recommendation = all_same
        ? "A3a/C2 can proceed with op_id keying as specified"
        : "Switch plan.ops keying to a more stable identifier (e.g. hash of (op_type, source-tensor names))";

    add(f, "runs",                std::to_string(runs.size()));
    add(f, "nodes_per_run",       std::to_string(runs[0].size()));
    add(f, "backend",             "cpu (ONEAPI_DEVICE_SELECTOR=opencl:cpu sidesteps m09zb)");
    add(f, "model_path",          mp);
    add(f, "n_ctx",               std::to_string(cparams.n_ctx));
    add(f, "n_ubatch",            std::to_string(cparams.n_ubatch));
    if (!all_same) {
        add(f, "first_divergence_run",   std::to_string(diverge_run));
        add(f, "first_divergence_index", std::to_string(diverge_index));
        if (diverge_index >= 0) {
            add(f, "first_divergence_run0",   diverge_a);
            add(f, "first_divergence_runN",   diverge_b);
        } else {
            add(f, "first_divergence_size_run0",  std::to_string(runs[0].size()));
            add(f, "first_divergence_size_runN",  std::to_string(runs[diverge_run].size()));
        }
    }
    add(f, "todo_multidevice",
        "Multi-device CPY-name stability check is the original D0.3 scope; "
        "preserve as future work once >=2 SYCL devices are usable "
        "on this host (B50 disabled per CLAUDE.md memory feedback_disable_b50.md) "
        "and E1 staging_buffer_pool fix (llama.cpp-m09zb) lands");
    add(f, "plan_section",
        "docs/plans/2026-04-24-planner-validation-phases.md Task 1");

    write_markdown(f, MD_PATH);
    write_json    (f, JSON_PATH);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return all_same ? 0 : 1;
}
