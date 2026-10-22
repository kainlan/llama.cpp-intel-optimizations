// test-moe-exec-policy: the MoE executor's knobs and its once-per-change [MOE-EXEC] log (moe-exec-policy.hpp;
// llama.cpp-36j0 Task 4). Host only: no device, no model, no SYCL.
#include "moe-exec-policy.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace ggml_sycl;

static int g_cases = 0, g_failures = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        ++g_cases;                                                             \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        }                                                                      \
    } while (0)

static std::map<std::string, std::string> g_env;

static const char * fake_getenv(const char * name) {
    auto it = g_env.find(name);
    return it == g_env.end() ? nullptr : it->second.c_str();
}

static void capture(void * ctx, const char * line) {
    static_cast<std::vector<std::string> *>(ctx)->push_back(line);
}

static moe_exec_decision decision(moe_exec_choice c, moe_exec_reason r = moe_exec_reason::CHEAPER) {
    moe_exec_decision d;
    d.choice    = c;
    d.reason    = r;
    d.p_touch   = 0.75;
    d.t_cpu_us  = 1234.0;
    d.t_ring_us = 2345.0;
    return d;
}

static size_t count_layer_lines(const std::vector<std::string> & lines) {
    size_t n = 0;
    for (const std::string & l : lines) {
        n += l.rfind("[MOE-EXEC] layer ", 0) == 0 ? 1 : 0;
    }
    return n;
}

int main() {
    // config: unset is AUTO, no trace, no bad value
    {
        g_env.clear();
        const moe_exec_config c = moe_exec_config_parse(fake_getenv);
        CHECK(c.policy == moe_exec_policy::AUTO, "unset policy is auto");
        CHECK(c.trace == 0, "unset trace is 0");
        CHECK(c.bad_policy == nullptr, "unset has no bad policy");
    }
    // config: each policy word
    {
        g_env.clear();
        g_env["GGML_SYCL_MOE_EXEC"] = "cpu";
        CHECK(moe_exec_config_parse(fake_getenv).policy == moe_exec_policy::CPU, "cpu parses");
        g_env["GGML_SYCL_MOE_EXEC"] = "ring";
        CHECK(moe_exec_config_parse(fake_getenv).policy == moe_exec_policy::RING, "ring parses");
        g_env["GGML_SYCL_MOE_EXEC"] = "auto";
        const moe_exec_config c     = moe_exec_config_parse(fake_getenv);
        CHECK(c.policy == moe_exec_policy::AUTO && c.bad_policy == nullptr, "auto parses, nothing rejected");
    }
    // config: a bad word is AUTO and is named once
    {
        g_env.clear();
        g_env["GGML_SYCL_MOE_EXEC"] = "gpu";
        const moe_exec_config c     = moe_exec_config_parse(fake_getenv);
        CHECK(c.policy == moe_exec_policy::AUTO, "bad policy falls back to auto");
        CHECK(c.bad_policy != nullptr && std::strcmp(c.bad_policy, "gpu") == 0, "bad policy names the value");
    }
    // config: trace is an integer level (0 off, 1 trace, 2 probe); junk and negatives are 0
    {
        g_env.clear();
        g_env["GGML_SYCL_MOE_EXEC_TRACE"] = "1";
        CHECK(moe_exec_config_parse(fake_getenv).trace == 1, "trace 1");
        g_env["GGML_SYCL_MOE_EXEC_TRACE"] = "2";
        CHECK(moe_exec_config_parse(fake_getenv).trace == 2, "trace 2");
        g_env["GGML_SYCL_MOE_EXEC_TRACE"] = "junk";
        CHECK(moe_exec_config_parse(fake_getenv).trace == 0, "junk trace is 0");
        g_env["GGML_SYCL_MOE_EXEC_TRACE"] = "-3";
        CHECK(moe_exec_config_parse(fake_getenv).trace == 0, "negative trace is 0");
    }
    // the ring_layers knob is removed: setting it changes nothing
    {
        g_env.clear();
        g_env["GGML_SYCL_MOE_RING_LAYERS"] = "5";
        const moe_exec_config c            = moe_exec_config_parse(fake_getenv);
        CHECK(c.policy == moe_exec_policy::AUTO && c.trace == 0, "RING_LAYERS is not read");
    }

    // log: 48 layers x 3 micro-batches, nothing changes after the first -> exactly 48 lines
    {
        std::vector<std::string> lines;
        moe_exec_decision_log    log;
        log.sink     = capture;
        log.sink_ctx = &lines;
        for (int ub = 0; ub < 3; ++ub) {
            for (int layer = 0; layer < 48; ++layer) {
                log.note(0, layer, decision(moe_exec_choice::RING), 1u << 20);
            }
        }
        CHECK(count_layer_lines(lines) == 48, "48 layers x 3 ubatches, no change: 48 lines");
        CHECK(lines[0].find("layer 0: ring (p=0.75 t_cpu=1234 us t_ring=2345 us cheaper)") != std::string::npos,
              "line carries the choice and the terms");
        CHECK(log.n_ring == 144 && log.n_cpu == 0, "every layer-ubatch is counted, not just the changes");
    }
    // log: one layer flips once -> 49 lines (positive control)
    {
        std::vector<std::string> lines;
        moe_exec_decision_log    log;
        log.sink     = capture;
        log.sink_ctx = &lines;
        for (int ub = 0; ub < 3; ++ub) {
            for (int layer = 0; layer < 48; ++layer) {
                const bool flip = layer == 7 && ub == 2;
                log.note(0, layer, decision(flip ? moe_exec_choice::CPU : moe_exec_choice::RING), 0);
            }
        }
        CHECK(count_layer_lines(lines) == 49, "one flip: 49 lines");
        CHECK(lines.back().find("layer 7: cpu") != std::string::npos, "the flip line names the new choice");
        CHECK(log.n_cpu == 1 && log.n_ring == 143, "counters follow the choices");
    }
    // log: devices keep separate state; forced and streamed bytes are counted; NOT_CONSULTED is ignored
    {
        std::vector<std::string> lines;
        moe_exec_decision_log    log;
        log.sink     = capture;
        log.sink_ctx = &lines;
        log.note(0, 3, decision(moe_exec_choice::RING, moe_exec_reason::FORCED), 3ull << 20);
        log.note(1, 3, decision(moe_exec_choice::RING, moe_exec_reason::FORCED), 5ull << 20);
        log.note(1, 3, moe_exec_decision(), 99);
        CHECK(count_layer_lines(lines) == 2, "same layer on two devices logs once each");
        CHECK(log.n_forced == 2, "forced counted");
        CHECK(log.bytes_streamed == (8ull << 20), "streamed bytes accumulate for RING only, NOT_CONSULTED ignored");
        char buf[160];
        log.summary(buf, sizeof(buf));
        CHECK(std::strcmp(buf, "[MOE-EXEC] summary: 2 layer-ubatches ring, 0 cpu, 2 forced, 8 MiB streamed") == 0,
              "summary format");
    }

    std::printf("%d cases, %d failures\n", g_cases, g_failures);
    return g_failures == 0 ? 0 : 1;
}
