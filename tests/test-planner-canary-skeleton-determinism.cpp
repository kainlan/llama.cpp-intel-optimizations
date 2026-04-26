// Canary D0.1 — skeleton graph size determinism.
//
// This canary is intentionally a thin orchestrator over
// test-mini-context-prototype. The prototype already implements the real
// proof protocol:
//   real-A: no_alloc=false, full weight load, real graph_reserve
//   real-B: no_alloc=false, second identical real reserve
//   mini:   no_alloc=true,use_mmap=false metadata-only mini-context
// PASS requires real-A == real-B and mini == real-A for compute-buffer sizes
// and FA auto-detection verdict.

#include "test-planner-canary-common.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace planner_canary;

static const char * MD_PATH   = "docs/plans/data/planner-canaries/d0.1-skeleton-determinism.md";
static const char * JSON_PATH = "tests/data/planner-canaries/d0.1.json";

struct model_case {
    std::string label;
    std::string path;
};

static std::string dirname_of(const char * path) {
    std::string  s   = path ? path : "";
    const size_t pos = s.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return s.substr(0, pos);
}

static int run_mini_context_proof(const std::string & proof_bin, const std::string & model_path) {
    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "[D0.1] fork failed: %s\n", std::strerror(errno));
        return 127;
    }

    if (pid == 0) {
        setenv("MISTRAL_PATH", model_path.c_str(), 1);
        execl(proof_bin.c_str(), proof_bin.c_str(), (char *) nullptr);
        std::fprintf(stderr, "[D0.1] execl failed for %s: %s\n", proof_bin.c_str(), std::strerror(errno));
        std::_Exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "[D0.1] waitpid failed: %s\n", std::strerror(errno));
        return 127;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 126;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    findings f;
    f.canary_id = "D0.1";
    f.result    = status::FAIL;

    const std::string proof_bin = dirname_of(argc > 0 ? argv[0] : "") + "/test-mini-context-prototype";
    add(f, "proof_binary", proof_bin);

    if (access(proof_bin.c_str(), X_OK) != 0) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "test-mini-context-prototype binary is not available next to D0.1 canary";
        f.recommendation = "Build the test-mini-context-prototype target and rerun D0.1";
        add(f, "errno", std::strerror(errno));
        write_markdown(f, MD_PATH);
        write_json(f, JSON_PATH);
        return 0;
    }

    const std::vector<model_case> models = {
        { "mistral_dense",  mistral_path() },
        { "gptoss_swa_moe", gptoss_path()  },
    };

    bool any_ran    = false;
    bool all_passed = true;

    for (const model_case & mc : models) {
        if (access(mc.path.c_str(), R_OK) != 0) {
            add(f, mc.label + "_status", "missing");
            add(f, mc.label + "_path", mc.path);
            continue;
        }

        std::fprintf(stderr, "[D0.1] running mini-context proof for %s: %s\n", mc.label.c_str(), mc.path.c_str());
        const int rc = run_mini_context_proof(proof_bin, mc.path);
        any_ran      = true;
        add(f, mc.label + "_path", mc.path);
        add(f, mc.label + "_exit", std::to_string(rc));
        if (rc != 0) {
            all_passed = false;
        }
    }

    if (!any_ran) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "No D0.1 model fixtures were available";
        f.recommendation = "Set MISTRAL_PATH and GPTOSS_PATH to readable GGUFs and rerun";
    } else if (all_passed) {
        f.result         = status::PASS;
        f.summary        = "Real/mini graph_reserve signals match for all available D0.1 model fixtures";
        f.recommendation = "A3a mini-context sizing remains validated for the tested dense and active SWA+MoE fixtures";
    } else {
        f.result  = status::FAIL;
        f.summary = "At least one real/mini graph_reserve proof failed";
        f.recommendation =
            "Inspect docs/plans/data/planner-canaries/mini-context-validation.md from the failing sub-proof";
    }

    add(f, "proof_protocol", "test-mini-context-prototype real-A/real-B/mini comparison");
    add(f, "plan_section", "docs/plans/2026-04-22-unified-memory-placement-plan.md D0.1");

    write_markdown(f, MD_PATH);
    write_json(f, JSON_PATH);

    return f.result == status::FAIL ? 1 : 0;
}
