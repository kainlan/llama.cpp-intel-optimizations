// Task 5 — Mini-context + FA auto-detect prototype.
//
// Question (from docs/plans/2026-04-24-planner-validation-phases.md §Task 5):
// Does a "throwaway mini-context" — built from metadata-only weights
// (`llama_model_params::no_alloc=true`) — produce per-backend buffer
// sizes and FA auto-detect decisions byte-identical to a real context
// on the same model + cparams?
//
// If yes, A3a's design — sizing the unified-cache zones from a
// throwaway pre-load mini-context — is mechanistically validated.
//
// Three workers, fork+exec'd from the driver:
//   real-A : no_alloc=false, full weight load, real graph_reserve
//   real-B : no_alloc=false, second identical run (determinism baseline)
//   mini   : no_alloc=true,  metadata-only load, graph_reserve from
//            shape-only graph
//
// PASS criteria:
//   1. real-A == real-B  (graph_reserve is deterministic across two
//      back-to-back contexts on the same model + cparams)
//   2. mini  == real-A   (mini-context yields the same per-backend
//      buffer sizes and FA decision as a real context)
//
// Signals captured per-worker (parsed from llama log output on stderr):
//   * Per-buft compute buffer size lines:
//       "<func>: <BUFT_NAME> compute buffer size = <N>.NN MiB"
//   * Flash-attention auto-detect verdict:
//       "<func>: Flash Attention was auto, set to enabled" / "...disabled"
//     (only emitted if cparams.flash_attn_type == AUTO)
//
// Backend: CPU (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`). The Task 5
// question is scheduler-level, not SYCL-specific, and CPU sidesteps
// the unrelated tiered_kv_buffer_clear wedge currently filed as
// llama.cpp-zhzbp on GPU.

#include "test-planner-canary-common.hpp"
#include "llama.h"
#include "common.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace planner_canary;

static const char * MD_PATH   = "docs/plans/data/planner-canaries/mini-context-validation.md";
static const char * JSON_PATH = "tests/data/planner-canaries/mini-context.json";

// Captured per-buft size, in bytes (we'll keep MiB to match the log
// precision). map preserves stable key ordering for byte-identity
// comparisons.
struct worker_signals {
    std::map<std::string, std::string> buft_sizes;  // buft_name -> "X.YZ MiB"
    std::string flash_attn_verdict;                  // "enabled", "disabled", or "" if not auto-resolved
    bool        ran  = false;
    int         crash_sig = 0;
};

// Parse llama log output.
//   "...: <buft> compute buffer size = <N>.NN MiB"
// We anchor on " compute buffer size = " to skip llama_context's own
// log prefix (varies with build flags).
static worker_signals parse_signals(const std::string & out) {
    worker_signals s;
    std::istringstream ss(out);
    std::string line;
    while (std::getline(ss, line)) {
        // "compute buffer size = " match
        size_t pos = line.find("compute buffer size = ");
        if (pos != std::string::npos) {
            // Walk left from pos to find the buft name (the token just
            // before "compute buffer size"). Tokens are separated by
            // whitespace; the buft name is the last whitespace-separated
            // word before the phrase.
            size_t end = pos;
            while (end > 0 && std::isspace((unsigned char) line[end - 1])) --end;
            size_t start = end;
            while (start > 0 && !std::isspace((unsigned char) line[start - 1])) --start;
            std::string buft = line.substr(start, end - start);

            std::string sz = line.substr(pos + std::strlen("compute buffer size = "));
            // Trim trailing whitespace
            while (!sz.empty() && std::isspace((unsigned char) sz.back())) sz.pop_back();

            if (!buft.empty() && !sz.empty()) {
                s.buft_sizes[buft] = sz;
                s.ran = true;
            }
            continue;
        }
        if (line.find("Flash Attention was auto, set to enabled") != std::string::npos) {
            s.flash_attn_verdict = "enabled";
            s.ran = true;
        } else if (line.find("Flash Attention was auto, set to disabled") != std::string::npos) {
            s.flash_attn_verdict = "disabled";
            s.ran = true;
        }
    }
    return s;
}

static std::string format_signals(const worker_signals & s) {
    std::ostringstream oss;
    oss << "fa=" << (s.flash_attn_verdict.empty() ? "(not-auto)" : s.flash_attn_verdict);
    for (const auto & p : s.buft_sizes) {
        oss << "; " << p.first << "=" << p.second;
    }
    return oss.str();
}

static bool signals_match(const worker_signals & a, const worker_signals & b,
                          std::string & first_divergence) {
    if (a.flash_attn_verdict != b.flash_attn_verdict) {
        first_divergence = "flash_attn_verdict differs: '" +
            a.flash_attn_verdict + "' vs '" + b.flash_attn_verdict + "'";
        return false;
    }
    if (a.buft_sizes.size() != b.buft_sizes.size()) {
        first_divergence = "buft count differs: " +
            std::to_string(a.buft_sizes.size()) + " vs " +
            std::to_string(b.buft_sizes.size());
        return false;
    }
    auto ia = a.buft_sizes.begin();
    auto ib = b.buft_sizes.begin();
    for (; ia != a.buft_sizes.end() && ib != b.buft_sizes.end(); ++ia, ++ib) {
        if (ia->first != ib->first) {
            first_divergence = "buft name differs: '" + ia->first + "' vs '" + ib->first + "'";
            return false;
        }
        if (ia->second != ib->second) {
            first_divergence = "buft '" + ia->first + "' size differs: " +
                ia->second + " vs " + ib->second;
            return false;
        }
    }
    return true;
}

// ============================================================
// Worker mode: load model, create context, exit. Llama log output
// goes to stderr, which the driver captures.
// ============================================================

static int run_worker(const char * model_path, const char * mode) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;  // CPU-only (sidesteps zhzbp on GPU)

    if (std::strcmp(mode, "mini") == 0) {
        mparams.no_alloc = true;
        // The mmap path asserts !no_alloc (llama-model.cpp:8686). Disable
        // mmap so the metadata-only load path is taken instead.
        mparams.use_mmap = false;
        std::fprintf(stderr, "[worker %s] no_alloc=true, use_mmap=false (mini-context, metadata-only)\n", mode);
    } else {
        std::fprintf(stderr, "[worker %s] no_alloc=false (real context)\n", mode);
    }

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "[worker %s] model load failed: %s\n", mode, model_path);
        return 2;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 1024;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    std::fprintf(stderr, "[worker %s] creating context (n_ctx=%u, ubatch=%u, fa=auto)\n",
                 mode, cparams.n_ctx, cparams.n_ubatch);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "[worker %s] context creation failed\n", mode);
        llama_model_free(model);
        return 3;
    }

    // Context ready. Signals (per-buft sizes + FA verdict) have been
    // logged to stderr during llama_init_from_model. Driver scrapes.
    std::fprintf(stderr, "[worker %s] context ready, exiting\n", mode);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ============================================================
// Driver mode
// ============================================================

struct worker_result {
    worker_signals signals;
    bool           crashed   = false;
    int            crash_sig = 0;
};

static worker_result run_worker_child(const char * self_path,
                                      const char * model_path,
                                      const char * mode) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        std::fprintf(stderr, "[driver] pipe() failed\n");
        return {};
    }

    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "[driver] fork() failed\n");
        close(pipefd[0]); close(pipefd[1]);
        return {};
    }
    if (pid == 0) {
        // Child: redirect stderr to pipe (llama logs go to stderr).
        // stdout left as-is for human-friendly progress reporting.
        close(pipefd[0]);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl(self_path, self_path, "--worker", model_path, mode, (char *) nullptr);
        std::perror("execl");
        std::_Exit(127);
    }

    // Parent: read child's stderr.
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

    worker_result r;
    r.signals = parse_signals(out);
    if (WIFSIGNALED(status)) {
        r.crashed   = true;
        r.crash_sig = WTERMSIG(status);
    }
    // Echo child stderr to driver stderr for diagnostic visibility
    std::fwrite(out.data(), 1, out.size(), stderr);
    return r;
}

static int run_driver(const char * self_path) {
    findings f;
    f.canary_id = "Task5";
    f.result    = status::FAIL;

    const std::string mp = mistral_path();
    if (access(mp.c_str(), R_OK) != 0) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "Mistral 7B GGUF unavailable at expected path";
        f.recommendation = "Set MISTRAL_PATH to a readable GGUF and re-run";
        add(f, "model_path", mp);
        write_markdown(f, MD_PATH);
        write_json    (f, JSON_PATH);
        return 0;
    }

    std::fprintf(stderr, "[driver] model: %s\n", mp.c_str());

    std::fprintf(stderr, "\n=== Worker 1: real-A (n_gpu_layers=0, no_alloc=false) ===\n");
    worker_result a = run_worker_child(self_path, mp.c_str(), "real-A");

    std::fprintf(stderr, "\n=== Worker 2: real-B (n_gpu_layers=0, no_alloc=false) ===\n");
    worker_result b = run_worker_child(self_path, mp.c_str(), "real-B");

    std::fprintf(stderr, "\n=== Worker 3: mini (n_gpu_layers=0, no_alloc=true) ===\n");
    worker_result m = run_worker_child(self_path, mp.c_str(), "mini");

    add(f, "model_path",   mp);
    add(f, "n_ctx",        "1024");
    add(f, "n_ubatch",     "512");
    add(f, "fa_type",      "AUTO");
    add(f, "backend",      "cpu (ONEAPI_DEVICE_SELECTOR=opencl:cpu sidesteps zhzbp on GPU)");
    add(f, "real_A_signals", format_signals(a.signals));
    add(f, "real_B_signals", format_signals(b.signals));
    add(f, "mini_signals",   format_signals(m.signals));

    if (a.crashed) add(f, "real_A_crash_sig", std::to_string(a.crash_sig));
    if (b.crashed) add(f, "real_B_crash_sig", std::to_string(b.crash_sig));
    if (m.crashed) add(f, "mini_crash_sig",   std::to_string(m.crash_sig));

    if (!a.signals.ran || !b.signals.ran) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "Real-context worker(s) did not emit any signals";
        f.recommendation = "Investigate llama_init_from_model failure on real-A/real-B";
        write_markdown(f, MD_PATH);
        write_json    (f, JSON_PATH);
        return 1;
    }

    std::string div_ab, div_am;
    bool ab_match = signals_match(a.signals, b.signals, div_ab);

    if (!ab_match) {
        f.result         = status::FAIL;
        f.summary        = "Two consecutive real contexts do NOT produce identical signals — graph_reserve is non-deterministic";
        f.recommendation = "A3a's mini-context approach blocked: real-context determinism is the prerequisite";
        add(f, "ab_first_divergence", div_ab);
        write_markdown(f, MD_PATH);
        write_json    (f, JSON_PATH);
        return 1;
    }

    add(f, "real_A_eq_real_B", "yes (graph_reserve is deterministic across two real contexts)");

    if (!m.signals.ran) {
        f.result         = status::FAIL;
        f.summary        = "Mini-context worker did not emit any signals — no_alloc=true context creation likely failed or skipped graph_reserve";
        f.recommendation = "Investigate `no_alloc=true` path through llama_init_from_model; may need a thin public accessor for size capture";
        if (m.crashed) add(f, "mini_crash_sig", std::to_string(m.crash_sig));
        write_markdown(f, MD_PATH);
        write_json    (f, JSON_PATH);
        return 1;
    }

    bool am_match = signals_match(a.signals, m.signals, div_am);

    if (am_match) {
        f.result = status::PASS;
        f.summary = "Mini-context (no_alloc=true) produces byte-identical buffer sizes + FA verdict to a real context — A3a approach validated";
        f.recommendation = "A3a can size unified-cache zones from a throwaway mini-context; A3b's FA auto-detect can run there too";
    } else {
        f.result = status::FAIL;
        f.summary = "Mini-context signals diverge from real-context signals — A3a needs a different sizing approach";
        f.recommendation = "Either capture sizes from a real context (defeats the throwaway-mini-context purpose) OR teach the planner to compensate for the divergence; document the specific delta below";
        add(f, "am_first_divergence", div_am);
    }

    write_markdown(f, MD_PATH);
    write_json    (f, JSON_PATH);
    return (f.result == status::PASS) ? 0 : 1;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    if (argc >= 4 && std::strcmp(argv[1], "--worker") == 0) {
        return run_worker(argv[2], argv[3]);
    }
    return run_driver(argv[0]);
}
