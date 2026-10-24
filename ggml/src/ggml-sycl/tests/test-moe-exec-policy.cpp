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
        CHECK(c.bad_policy[0] == '\0', "unset has no bad policy");
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
        CHECK(c.policy == moe_exec_policy::AUTO && c.bad_policy[0] == '\0', "auto parses, nothing rejected");
    }
    // config: a bad word is AUTO and is named once
    {
        g_env.clear();
        g_env["GGML_SYCL_MOE_EXEC"] = "gpu";
        const moe_exec_config c     = moe_exec_config_parse(fake_getenv);
        CHECK(c.policy == moe_exec_policy::AUTO, "bad policy falls back to auto");
        CHECK(std::strcmp(c.bad_policy, "gpu") == 0, "bad policy names the value");
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
        g_env["GGML_SYCL_MOE_EXEC_TRACE"] = "7";
        CHECK(moe_exec_config_parse(fake_getenv).trace == 2, "trace above 2 clamps to 2");
        g_env["GGML_SYCL_MOE_EXEC_TRACE"] = "0";
        CHECK(moe_exec_config_parse(fake_getenv).trace == 0, "trace 0");
        g_env["GGML_SYCL_MOE_EXEC_TRACE"] = "";
        CHECK(moe_exec_config_parse(fake_getenv).trace == 0, "empty trace is 0");
        g_env["GGML_SYCL_MOE_EXEC_TRACE"] = "-3";
        CHECK(moe_exec_config_parse(fake_getenv).trace == 0, "negative trace is 0");
    }
    // log: 48 layers x 3 micro-batches, nothing changes after the first -> exactly 48 lines
    {
        std::vector<std::string> lines;
        moe_exec_decision_log    log(capture, &lines);
        for (int ub = 0; ub < 3; ++ub) {
            for (int layer = 0; layer < 48; ++layer) {
                log.note(0, layer, decision(moe_exec_choice::RING));
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
        moe_exec_decision_log    log(capture, &lines);
        for (int ub = 0; ub < 3; ++ub) {
            for (int layer = 0; layer < 48; ++layer) {
                const bool flip = layer == 7 && ub == 2;
                log.note(0, layer, decision(flip ? moe_exec_choice::CPU : moe_exec_choice::RING));
            }
        }
        CHECK(count_layer_lines(lines) == 49, "one flip: 49 lines");
        CHECK(lines.back().find("layer 7: cpu") != std::string::npos, "the flip line names the new choice");
        CHECK(log.n_cpu == 1 && log.n_ring == 143, "counters follow the choices");
    }
    // log: the trace level adds one INFO-side line per consulted layer-ubatch and leaves the change lines alone
    for (int level = 0; level <= 2; ++level) {
        std::vector<std::string> lines;
        moe_exec_decision_log    log(capture, &lines, level, capture);
        for (int ub = 0; ub < 3; ++ub) {
            for (int layer = 0; layer < 48; ++layer) {
                log.note(0, layer, decision(moe_exec_choice::RING));
            }
        }
        log.note(0, 0, moe_exec_decision());  // NOT_CONSULTED: no line of either kind
        size_t n_trace = 0;
        for (const std::string & l : lines) {
            n_trace += l.rfind("[MOE-EXEC-TRACE] dev 0 layer ", 0) == 0 ? 1 : 0;
        }
        CHECK(count_layer_lines(lines) == 48, "trace level does not change the 48 change lines");
        CHECK(n_trace == (level == 0 ? 0u : 144u),
              "trace lines: none at level 0, one per layer-ubatch at level 1 and 2");
        CHECK(level == 0 || lines[0].find("[MOE-EXEC-TRACE] dev 0 layer 0: ring (p=0.75") == 0,
              "the trace line leads the change line and carries the terms");
    }
    // log: a trace level with no trace sink is silent
    {
        std::vector<std::string> lines;
        moe_exec_decision_log    log(capture, &lines, 1, nullptr);
        log.note(0, 0, decision(moe_exec_choice::RING));
        CHECK(lines.size() == 1, "trace level without a trace sink prints only the change line");
    }
    // log: devices keep separate state; forced and streamed bytes are counted; NOT_CONSULTED is ignored
    {
        std::vector<std::string> lines;
        moe_exec_decision_log    log(capture, &lines);
        log.note(0, 3, decision(moe_exec_choice::RING, moe_exec_reason::FORCED));
        log.note(1, 3, decision(moe_exec_choice::RING, moe_exec_reason::FORCED));
        log.note(1, 3, moe_exec_decision());
        CHECK(count_layer_lines(lines) == 2, "same layer on two devices logs once each");
        CHECK(log.n_forced == 2, "forced counted");
        CHECK(log.bytes_streamed == 0, "a layer note carries no bytes: the windows' bytes are the streamed total");
        char buf[160];
        CHECK(log.take_summary(buf, sizeof(buf)), "a consulted log gives a summary");
        CHECK(std::strcmp(buf,
                          "[MOE-EXEC] summary: 2 layer-ubatches ring, 0 cpu, 2 forced, 0 MiB streamed, 0 windows, "
                          "0 exposed, 0 misses") == 0,
              "summary format");
        CHECK(!log.take_summary(buf, sizeof(buf)), "a second take gives nothing");
        log.note(0, 3, decision(moe_exec_choice::RING, moe_exec_reason::FORCED));
        log.note_window(/*exposed*/ true, /*misses*/ 5, 0);  // an exposed window's misses are not counted
        log.note_window(false, 0, 0);
        log.note_window(false, 2, 0);
        CHECK(log.take_summary(buf, sizeof(buf)) &&
                  std::strcmp(buf,
                              "[MOE-EXEC] summary: 1 layer-ubatches ring, 0 cpu, 1 forced, 0 MiB streamed, "
                              "3 windows, 1 exposed, 2 misses") == 0,
              "window counters in the summary");
        CHECK(!log.take_summary(buf, sizeof(buf)), "a take with nothing consulted prints nothing");
        log.note(0, 4, decision(moe_exec_choice::CPU));
        CHECK(log.take_summary(buf, sizeof(buf)) &&
                  std::strcmp(buf,
                              "[MOE-EXEC] summary: 0 layer-ubatches ring, 1 cpu, 0 forced, 0 MiB streamed, 0 windows, "
                              "0 exposed, 0 misses") == 0,
              "a note after a take counts from zero, window counters included (the reset witness)");
    }

    // log: MiB streamed is the sum of the bytes each window staged (its own and its prefetch's), summed before the
    // shift: two windows of 3.5 and 4.5 MiB are 8 MiB, where shifting each would read 7
    {
        std::vector<std::string> lines;
        moe_exec_decision_log    log(capture, &lines);
        log.note(0, 3, decision(moe_exec_choice::RING));
        log.note_window(/*exposed*/ true, 4, (7ull << 20) / 2);
        log.note_window(false, 1, (9ull << 20) / 2);
        log.note_window(false, 0, 0);  // a window the last call prefetched stages nothing of its own
        char buf[160];
        CHECK(log.take_summary(buf, sizeof(buf)) &&
                  std::strcmp(buf,
                              "[MOE-EXEC] summary: 1 layer-ubatches ring, 0 cpu, 0 forced, 8 MiB streamed, "
                              "3 windows, 1 exposed, 1 misses") == 0,
              "MiB streamed is the sum of the windows' staged bytes, two windows of different byte counts");
        log.note(0, 3, decision(moe_exec_choice::RING));
        CHECK(log.take_summary(buf, sizeof(buf)) && std::strstr(buf, "0 MiB streamed") != nullptr,
              "the streamed bytes reset with the summary");
    }

    std::printf("%d cases, %d failures\n", g_cases, g_failures);
    return g_failures == 0 ? 0 : 1;
}
