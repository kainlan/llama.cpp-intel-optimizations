// Host test for the host pool's inference-phase gate (rulings §M77, narrowed by
// §M9a): grow_zone() and grow_into() skip the PP/TG-phase warning only for a
// thread holding a TRANSACTION-kind L0 token.  A LOAD or LIFECYCLE token is not
// exempt, and neither is no token at all.  Each warning names the calling
// `site=`, threaded through grow().
//
// The pool is a real pinned_chunk_pool on the OpenCL CPU device with 1 MB chunks
// (GGML_SYCL_PINNED_CHUNK_MB), so a runtime allocation, a zone configuration and a
// zone growth each add a real chunk.  The assert mode (GGML_SYCL_HOST_ALLOC_PHASE_GATE=2)
// is read once per process, so its arms re-exec this binary.
//
// Nothing here touches a GPU: the registration pins the selector to the CPU.
//
// Usage:
//   ./build/bin/test-sycl-pool-phase-gate                    # every case
//   ./build/bin/test-sycl-pool-phase-gate <child> <kind>     # one assert-mode child

#include "ggml.h"
#include "pinned-pool.hpp"
#include "unified-cache.hpp"

#include <sys/wait.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sycl/sycl.hpp>

using namespace ggml_sycl;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

constexpr size_t MB = 1024ULL * 1024ULL;

std::string g_warn;

void capture_log(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN && std::strstr(text, "[HOST-POOL]") != nullptr) {
        g_warn += text;
    }
}

// One growth per `what`, each through a different site of the pool.
void grow_at(pinned_chunk_pool & pool, const char * what, offload_phase phase) {
    if (std::strcmp(what, "allocate_from_chunks") == 0) {
        (void) pool.allocate_runtime(64 * 1024);
    } else if (std::strcmp(what, "allocate_segmented") == 0) {
        (void) pool.allocate_segmented(64 * 1024);
    } else if (std::strcmp(what, "pre_allocate") == 0) {
        (void) pool.pre_allocate(1 * MB);
    } else if (std::strcmp(what, "pre_allocate_runtime") == 0) {
        (void) pool.pre_allocate_runtime_chunks(1 * MB);
    } else if (std::strcmp(what, "configure_zones") == 0) {
        pool.configure_zones(512 * 1024, 64 * 1024, 64 * 1024, 64 * 1024);
    } else if (std::strcmp(what, "grow_zone") == 0) {
        // The zones are configured outside the phase: only grow_zone() is under test.
        offload_stats_set_phase(offload_phase::UNKNOWN);
        pool.configure_zones(64 * 1024, 64 * 1024, 64 * 1024, 64 * 1024);
        offload_stats_set_phase(phase);
        (void) pool.grow_zone(host_zone_id::WEIGHT, 512 * 1024);
    }
}

const char * const kSites[] = { "allocate_from_chunks", "allocate_segmented", "pre_allocate",
                                "pre_allocate_runtime", "configure_zones",    "grow_zone" };

// Run one growth in `phase` under an optional token and return the gate's warnings.
std::string warnings_for(const char * site, offload_phase phase, int token_kind /* -1: none */) {
    sycl::queue       q{ sycl::cpu_selector_v };
    pinned_chunk_pool pool(q, 256 * MB);
    g_warn.clear();
    offload_stats_set_phase(phase);
    if (token_kind < 0) {
        grow_at(pool, site, phase);
    } else {
        ggml_sycl_replan_token token(static_cast<ggml_sycl_replan_kind>(token_kind));
        grow_at(pool, site, phase);
    }
    offload_stats_set_phase(offload_phase::UNKNOWN);
    return g_warn;
}

void test_in_process() {
    for (const char * site : kSites) {
        const std::string name = site;
        const std::string none = warnings_for(site, offload_phase::TG, -1);
        CHECK(none.find("[HOST-POOL]") != std::string::npos, (name + ": TG, no token warns").c_str());
        CHECK(none.find(std::string("site=") + site) != std::string::npos,
              (name + ": the warning names its site").c_str());
        CHECK(none.find("during tg phase") != std::string::npos, (name + ": the warning names the phase").c_str());

        const std::string txn = warnings_for(site, offload_phase::TG, GGML_SYCL_REPLAN_KIND_TRANSACTION);
        CHECK(txn.empty(), (name + ": a TRANSACTION token skips the gate").c_str());
        const std::string load = warnings_for(site, offload_phase::TG, GGML_SYCL_REPLAN_KIND_LOAD);
        CHECK(load.find("[HOST-POOL]") != std::string::npos, (name + ": a LOAD token is not exempt").c_str());
        const std::string life = warnings_for(site, offload_phase::TG, GGML_SYCL_REPLAN_KIND_LIFECYCLE);
        CHECK(life.find("[HOST-POOL]") != std::string::npos, (name + ": a LIFECYCLE token is not exempt").c_str());

        // Controls: the gate is a PP/TG gate, and PP counts.
        const std::string pp = warnings_for(site, offload_phase::PP, -1);
        CHECK(pp.find("during pp phase") != std::string::npos, (name + ": PP, no token warns").c_str());
        const std::string load_phase = warnings_for(site, offload_phase::LOAD, -1);
        CHECK(load_phase.empty(), (name + ": the load phase is not gated").c_str());
        const std::string unknown = warnings_for(site, offload_phase::UNKNOWN, -1);
        CHECK(unknown.empty(), (name + ": the unknown phase is not gated").c_str());
    }
    // A nested TRANSACTION under a TRANSACTION skips; a LIFECYCLE inside it keeps
    // the outermost kind, so the exemption holds.
    {
        sycl::queue       q{ sycl::cpu_selector_v };
        pinned_chunk_pool pool(q, 256 * MB);
        g_warn.clear();
        offload_stats_set_phase(offload_phase::TG);
        {
            ggml_sycl_replan_token outer(GGML_SYCL_REPLAN_KIND_TRANSACTION);
            ggml_sycl_replan_token inner(GGML_SYCL_REPLAN_KIND_LIFECYCLE);
            (void) pool.allocate_runtime(64 * 1024);
        }
        offload_stats_set_phase(offload_phase::UNKNOWN);
        CHECK(g_warn.empty(), "a LIFECYCLE hold nested under a TRANSACTION keeps the exemption");
    }
}

int child(const char * site, int token_kind) {
    const std::string w = warnings_for(site, offload_phase::TG, token_kind);
    std::printf("returned without aborting: %zu bytes of warning\n", w.size());
    return 0;
}

bool run_child(const char * self, const char * site, int token_kind, std::string & out, int & status) {
    const std::string cmd = std::string(
                                "GGML_NO_BACKTRACE=1 GGML_SYCL_HOST_ALLOC_PHASE_GATE=2 "
                                "GGML_SYCL_PINNED_CHUNK_MB=1 ") +
                            self + " " + site + " " + std::to_string(token_kind) + " 2>&1";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) {
        return false;
    }
    char buf[512];
    while (std::fgets(buf, sizeof(buf), p)) {
        out += buf;
    }
    status = pclose(p);
    return true;
}

void test_assert_mode(const char * self) {
    struct arm {
        const char * site;
        const char * regex_text;
    };

    const arm arms[] = {
        { "allocate_from_chunks", "host pool chunk allocation during inference" },
        { "configure_zones",      "host pool chunk allocation during inference" },
        { "grow_zone",            "host pool zone growth during inference"      },
    };
    for (const arm & a : arms) {
        const std::string name = a.site;
        std::string       out;
        int               status = 0;
        CHECK(run_child(self, a.site, -1, out, status), "assert child started");
        const bool died = (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) ||
                          (WIFEXITED(status) && WEXITSTATUS(status) == 128 + SIGABRT);
        CHECK(died, (name + ": no token in assert mode aborts").c_str());
        CHECK(out.find(a.regex_text) != std::string::npos, (name + ": the abort carries its message").c_str());
        CHECK(out.find("returned without aborting") == std::string::npos,
              (name + ": nothing ran past the gate").c_str());

        for (const int kind : { (int) GGML_SYCL_REPLAN_KIND_LOAD, (int) GGML_SYCL_REPLAN_KIND_LIFECYCLE }) {
            std::string o2;
            int         s2 = 0;
            CHECK(run_child(self, a.site, kind, o2, s2), "assert child started");
            CHECK(o2.find(a.regex_text) != std::string::npos,
                  (name + ": a " + ggml_sycl_replan_kind_name((ggml_sycl_replan_kind) kind) +
                   " token still aborts in assert mode")
                      .c_str());
        }

        std::string o3;
        int         s3 = 0;
        CHECK(run_child(self, a.site, (int) GGML_SYCL_REPLAN_KIND_TRANSACTION, o3, s3), "assert child started");
        CHECK(s3 == 0 && o3.find("returned without aborting") != std::string::npos,
              (name + ": a TRANSACTION token passes in assert mode").c_str());
        if (s3 != 0) {
            std::fprintf(stderr, "%s printed:\n%s", a.site, o3.c_str());
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_log_set(capture_log, nullptr);
    if (argc > 2) {
        return child(argv[1], std::atoi(argv[2]));
    }
    // The arms below use 1 MB chunks, set before the first pool is built.
    setenv("GGML_SYCL_PINNED_CHUNK_MB", "1", 1);
    test_in_process();
    test_assert_mode(argv[0]);
    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-pool-phase-gate: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-pool-phase-gate: all ok\n");
    return 0;
}
