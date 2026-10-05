// llama.cpp-z5fn: [HOSTMEM] cost model. Pure host test, no model, no GPU.
//
// The default (no GGML_SYCL_HOSTMEM) must be two cheap lines per run and must
// never select the expensive readers (smaps, mallinfo2); the full mode selects
// everything. host_mem_plan_for() is the single decision point, so pinning it
// here pins what the default path may do.

#include "../ggml/src/ggml-sycl/host-mem-ledger.hpp"

#include <cstdio>

using ggml_sycl::host_mem_phase;
using ggml_sycl::host_mem_plan;
using ggml_sycl::host_mem_plan_for;

static int g_failures = 0;

static void expect(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

int main() {
    const host_mem_phase phases[] = { host_mem_phase::LOAD_END, host_mem_phase::PP_TO_TG_BEFORE,
                                      host_mem_phase::PP_TO_TG_AFTER, host_mem_phase::LADDER };

    // The cheap-by-default property, over every phase and both first/repeat states.
    for (host_mem_phase ph : phases) {
        for (int first = 0; first < 2; ++first) {
            const host_mem_plan p = host_mem_plan_for(ph, /*full_mode=*/false, first != 0);
            expect(!p.full, "default mode never selects the expensive readers");
            expect(!p.rate_limited, "default mode needs no rate limit: it emits at most once per phase");
        }
    }

    // Default emits exactly: load end, and the FIRST PP->TG only.
    expect(host_mem_plan_for(host_mem_phase::LOAD_END, false, false).emit, "default: load-end line");
    expect(host_mem_plan_for(host_mem_phase::PP_TO_TG_BEFORE, false, true).emit, "default: first PP->TG line");
    expect(!host_mem_plan_for(host_mem_phase::PP_TO_TG_BEFORE, false, false).emit, "default: repeat PP->TG is silent");
    expect(!host_mem_plan_for(host_mem_phase::PP_TO_TG_AFTER, false, true).emit, "default: no after-refresh line");
    expect(!host_mem_plan_for(host_mem_phase::LADDER, false, false).emit, "default: no graph_compute ladder");

    // Positive control: full mode is where the expensive readers and the ladder live,
    // so the default assertions above are not satisfied by a plan that never says full.
    for (host_mem_phase ph : phases) {
        const host_mem_plan p = host_mem_plan_for(ph, /*full_mode=*/true, false);
        expect(p.emit && p.full, "full mode emits every phase with the expensive readers");
    }
    expect(host_mem_plan_for(host_mem_phase::PP_TO_TG_BEFORE, true, false).rate_limited,
           "full mode rate-limits the repeating PP->TG line");
    expect(!host_mem_plan_for(host_mem_phase::LOAD_END, true, false).rate_limited, "load-end is not rate limited");

    if (g_failures == 0) {
        std::printf("PASS\n");
    }
    return g_failures == 0 ? 0 : 1;
}
