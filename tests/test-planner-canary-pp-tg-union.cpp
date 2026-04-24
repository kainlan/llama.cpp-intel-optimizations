// Canary D0.2 — PP + TG graph union.
// Builds graphs at PP-shape (ubatch=max) and TG-shape (ubatch=1) for the
// same model, collects op sets from each, verifies the union covers every
// op that either will execute. Ops are collected by installing an eval
// callback on the scheduler (llama_context_params::cb_eval) which ggml
// calls for every node *before* the per-split compute pass, so the entire
// op set is captured even if compute itself later aborts.
//
// Execution model:
//   * Parent process ("driver"): reads any existing partial findings
//     from tests/data/planner-canaries/d0.2.json, fork+execs itself
//     repeatedly as --worker <model> <shape> children, merges each
//     child's stdout op list into findings, writes final result.
//   * Worker process: loads one model, creates one context at the given
//     shape, runs one decode, prints the collected op names to stdout
//     as "OPS: <comma-separated>" (plus ASK_CALLS: <n>), then exits.
//     Backend aborts in the worker do not affect the driver.
//
// The two-phase design survives backend abort()s during decode (e.g.
// CPU vec_dot_f16 isnan assertion when flash attention softmax produces
// NaN on a BOS-only prompt, or SYCL DEVICE_LOST on a wedged GPU). The
// op set captured before the abort is still printed by the worker because
// cb_eval runs for every node *before* that split's compute call.

#include "test-planner-canary-common.hpp"
#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace planner_canary;

static const char * MD_PATH   = "docs/plans/data/planner-canaries/d0.2-pp-tg-union.md";
static const char * JSON_PATH = "tests/data/planner-canaries/d0.2.json";

// ============================================================
// Worker mode (--worker <model_path> <shape>)
// ============================================================

struct collect_ctx {
    std::set<std::string> ops;
    size_t                ask_calls = 0;
};

// Worker-global pointer so the signal handler can spill the partial op
// set to stdout on abort.
static collect_ctx * g_worker_ctx = nullptr;

// Pre-rendered stdout payload. We build this incrementally from
// op_collect_cb so the signal handler only needs to write() a buffer
// that's already in memory — no STL allocations, no stdio buffering
// races with whatever was in flight when the signal fired.
static std::string g_worker_payload;
static bool        g_worker_spilled = false;  // idempotent guard

static void worker_rebuild_payload(collect_ctx * ctx) {
    std::ostringstream oss;
    oss << "ASK_CALLS: " << ctx->ask_calls << "\n";
    oss << "OPS: ";
    bool first = true;
    for (const auto & op : ctx->ops) {
        if (!first) oss << ",";
        oss << op;
        first = false;
    }
    oss << "\n";
    g_worker_payload = oss.str();
}

static void worker_spill_and_die(int sig) {
    if (!g_worker_spilled && g_worker_ctx) {
        g_worker_spilled = true;
        // Write the pre-rendered payload (OPS + ASK_CALLS lines) plus
        // a CRASHED line. write(2) is async-signal-safe; printf is not.
        (void)write(STDOUT_FILENO, g_worker_payload.data(), g_worker_payload.size());
        char tail[32];
        int n = std::snprintf(tail, sizeof(tail), "CRASHED: %d\n", sig);
        if (n > 0) (void)write(STDOUT_FILENO, tail, (size_t)n);
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

static bool op_collect_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = static_cast<collect_ctx *>(user_data);
    if (!ask) {
        return true;
    }
    ctx->ask_calls++;
    bool new_op = false;
    if (t && t->op != GGML_OP_NONE) {
        auto inserted = ctx->ops.insert(ggml_op_name(t->op));
        new_op = inserted.second;
    }
    // Refresh the crash-handler payload whenever the op set changes
    // (to include the newly-seen op) or every 64 asks (to keep the
    // ask_calls counter close to current). The new-op path is the
    // only one that affects the canary's PASS/FAIL rubric; the counter
    // is just diagnostic.
    if (new_op || (ctx->ask_calls & 63) == 0) {
        worker_rebuild_payload(ctx);
    }
    return false;
}

static int run_worker(const char * model_path, const char * shape) {
    std::signal(SIGABRT, worker_spill_and_die);
    std::signal(SIGSEGV, worker_spill_and_die);
    std::signal(SIGFPE,  worker_spill_and_die);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;  // CPU-only path is sufficient for op topology
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "[worker %s] model load failed: %s\n", shape, model_path);
        return 2;
    }

    collect_ctx cc;
    g_worker_ctx = &cc;

    // n_ctx kept tight so the SYCL planner's conservative KV sizing
    // doesn't blow the budget; we only need room for the 512 PP tokens.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                  = 1024;
    cparams.cb_eval                = op_collect_cb;
    cparams.cb_eval_user_data      = &cc;

    int32_t n_tokens;
    if (std::strcmp(shape, "pp") == 0) {
        cparams.n_batch  = 512;
        cparams.n_ubatch = 512;
        n_tokens         = 512;
    } else {
        cparams.n_batch  = 1;
        cparams.n_ubatch = 1;
        n_tokens         = 1;
    }

    std::fprintf(stderr, "[worker %s] creating context (n_ctx=%u, ubatch=%u)\n",
                 shape, cparams.n_ctx, cparams.n_ubatch);
    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "[worker %s] context creation failed\n", shape);
        llama_model_free(model);
        return 3;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));
    for (int i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, bos, i, {0}, i == n_tokens - 1);
    }

    std::fprintf(stderr, "[worker %s] calling llama_decode (%d tokens)\n", shape, n_tokens);
    int rc = llama_decode(ctx, batch);
    std::fprintf(stderr, "[worker %s] llama_decode returned %d\n", shape, rc);

    // Decode path completed (or was about to crash during teardown).
    // Disarm the crash-spill handlers so a later SIGABRT in llama_free
    // doesn't duplicate our stdout output.
    g_worker_ctx = nullptr;
    std::signal(SIGABRT, SIG_DFL);
    std::signal(SIGSEGV, SIG_DFL);
    std::signal(SIGFPE,  SIG_DFL);

    // Normal path: decode has reached a point where cb_eval ran for
    // every node. Print ops to stdout for the driver.
    std::fprintf(stdout, "ASK_CALLS: %zu\n", cc.ask_calls);
    std::fprintf(stdout, "OPS: ");
    bool first = true;
    for (const auto & op : cc.ops) {
        if (!first) std::fprintf(stdout, ",");
        std::fprintf(stdout, "%s", op.c_str());
        first = false;
    }
    std::fprintf(stdout, "\n");
    std::fprintf(stdout, "CRASHED: 0\n");
    std::fflush(stdout);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ============================================================
// Driver mode (default, no --worker)
// ============================================================

struct worker_result {
    std::set<std::string> ops;
    size_t                ask_calls = 0;
    bool                  crashed   = false;
    int                   crash_sig = 0;
    bool                  ran       = false;
};

// Parse a worker's stdout for ASK_CALLS, OPS, CRASHED lines.
static worker_result parse_worker_output(const std::string & out) {
    worker_result r;
    std::istringstream ss(out);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.rfind("ASK_CALLS: ", 0) == 0) {
            r.ask_calls = std::strtoull(line.c_str() + 11, nullptr, 10);
            r.ran = true;
        } else if (line.rfind("OPS: ", 0) == 0) {
            std::string ops = line.substr(5);
            std::string token;
            std::istringstream ops_ss(ops);
            while (std::getline(ops_ss, token, ',')) {
                if (!token.empty()) r.ops.insert(token);
            }
            r.ran = true;
        } else if (line.rfind("CRASHED: ", 0) == 0) {
            int sig = std::atoi(line.c_str() + 9);
            r.crashed   = sig != 0;
            r.crash_sig = sig;
        }
    }
    return r;
}

// Fork+exec a worker child. Child's stdout is captured and returned.
static worker_result run_worker_child(const char * self_path,
                                      const char * model_path,
                                      const char * shape) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        fprintf(stderr, "[driver] pipe() failed\n");
        return {};
    }

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[driver] fork() failed\n");
        close(pipefd[0]); close(pipefd[1]);
        return {};
    }
    if (pid == 0) {
        // Child: redirect stdout to pipe, exec self as worker.
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        execl(self_path, self_path, "--worker", model_path, shape, (char *)nullptr);
        std::perror("execl");
        std::_Exit(127);
    }

    // Parent: read child's stdout.
    close(pipefd[1]);
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        out.append(buf, buf + n);
    }
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    worker_result r = parse_worker_output(out);
    if (!r.crashed && WIFSIGNALED(status)) {
        r.crashed   = true;
        r.crash_sig = WTERMSIG(status);
    }
    return r;
}

static std::string join_set(const std::set<std::string> & s) {
    std::ostringstream oss;
    bool first = true;
    for (const auto & x : s) {
        if (!first) oss << ",";
        oss << x;
        first = false;
    }
    return oss.str();
}

static std::string model_label(const std::string & path) {
    auto slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static int run_driver(const char * self_path) {
    findings f;
    f.canary_id = "D0.2";
    f.result    = status::FAIL;

    const std::string mistral = mistral_path();
    const std::string gptoss  = gptoss_path();

    std::vector<std::string> models;
    if (access(mistral.c_str(), R_OK) == 0) models.push_back(mistral);
    if (access(gptoss.c_str(),  R_OK) == 0) models.push_back(gptoss);

    if (models.empty()) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "No test models available";
        f.recommendation = "set MISTRAL_PATH or GPTOSS_PATH env vars to a GGUF file";
        write_markdown(f, MD_PATH);
        write_json    (f, JSON_PATH);
        return 0;
    }

    bool any_collection_empty = false;
    bool any_shape_specific   = false;

    for (const auto & mp : models) {
        const std::string label = model_label(mp);
        fprintf(stderr, "[driver] %s: running PP worker\n", label.c_str());
        worker_result pp = run_worker_child(self_path, mp.c_str(), "pp");
        fprintf(stderr, "[driver] %s: PP ask_calls=%zu ops=%zu crashed=%d sig=%d\n",
                label.c_str(), pp.ask_calls, pp.ops.size(), pp.crashed, pp.crash_sig);

        fprintf(stderr, "[driver] %s: running TG worker\n", label.c_str());
        worker_result tg = run_worker_child(self_path, mp.c_str(), "tg");
        fprintf(stderr, "[driver] %s: TG ask_calls=%zu ops=%zu crashed=%d sig=%d\n",
                label.c_str(), tg.ask_calls, tg.ops.size(), tg.crashed, tg.crash_sig);

        std::set<std::string> pp_only, tg_only;
        for (const auto & o : tg.ops) if (!pp.ops.count(o)) tg_only.insert(o);
        for (const auto & o : pp.ops) if (!tg.ops.count(o)) pp_only.insert(o);

        add(f, label + ":pp_ask_calls", std::to_string(pp.ask_calls));
        add(f, label + ":tg_ask_calls", std::to_string(tg.ask_calls));
        add(f, label + ":pp_ops_count", std::to_string(pp.ops.size()));
        add(f, label + ":tg_ops_count", std::to_string(tg.ops.size()));
        add(f, label + ":pp_ops",       join_set(pp.ops));
        add(f, label + ":tg_ops",       join_set(tg.ops));
        add(f, label + ":pp_only_ops",  join_set(pp_only));
        add(f, label + ":tg_only_ops",  join_set(tg_only));
        if (pp.crashed) {
            add(f, label + ":pp_decode_crash_sig", std::to_string(pp.crash_sig));
        }
        if (tg.crashed) {
            add(f, label + ":tg_decode_crash_sig", std::to_string(tg.crash_sig));
        }

        if (pp.ops.empty() || tg.ops.empty()) {
            any_collection_empty = true;
        }
        if (!pp_only.empty() || !tg_only.empty()) {
            any_shape_specific = true;
        }
    }

    // Result rubric:
    //   INCONCLUSIVE: a worker crashed AND produced no ops (no usable data).
    //   FAIL: ops collected but one set ended up empty for some reason.
    //   PASS: both PP and TG op sets were enumerated. A3a must run
    //         graph_reserve at BOTH shapes iff any model has a non-empty
    //         pp_only or tg_only.
    if (any_collection_empty) {
        f.result         = status::FAIL;
        f.summary        = "Op collection failed for one or both shapes";
        f.recommendation = "Repair the op-collection helper before A3a proceeds";
    } else {
        f.result = status::PASS;
        if (any_shape_specific) {
            f.summary        = "PP and TG produce distinct op sets; double-reserve + union is required";
            f.recommendation = "A3a MUST run graph_reserve at both ubatch=max and ubatch=1 and union the plan.ops tables";
        } else {
            f.summary        = "PP and TG produce identical op sets across tested models; single-shape reserve suffices";
            f.recommendation = "A3a can size from a single shape";
        }
    }

    write_markdown(f, MD_PATH);
    write_json    (f, JSON_PATH);
    return (f.result == status::PASS) ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "--worker") == 0) {
        return run_worker(argv[2], argv[3]);
    }
    return run_driver(argv[0]);
}
