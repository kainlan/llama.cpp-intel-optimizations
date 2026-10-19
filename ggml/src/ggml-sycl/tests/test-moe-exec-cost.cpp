// test-moe-exec-cost: the MoE executor's per-layer cost (moe-exec-cost.hpp; llama.cpp-36j0) on the measured shapes.
// Host only: no device, no model, no SYCL.
#include "moe-exec-cost.hpp"

#include <cmath>
#include <cstdio>

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

static moe_exec_rates b70_rates() {
    moe_exec_rates r;
    r.h2d_gbps     = 26.5;
    r.cpu_gbps     = 22.0;
    r.reorder_gbps = 200.0;
    r.dev_gflops   = 50000.0;
    r.cpu_gflops   = 1000.0;
    r.cpu_fixed_us = 500.0;
    r.dev_fixed_us = 800.0;
    return r;
}

static moe_exec_rates b50_rates() {
    moe_exec_rates r = b70_rates();
    r.h2d_gbps       = 27.1;
    r.cpu_gbps       = 5.6;
    r.dev_gflops     = 25000.0;
    return r;
}

static moe_exec_layer_inputs qwen_layer(uint32_t T) {
    moe_exec_layer_inputs in;
    in.n_tokens          = T;
    in.n_expert          = 512;
    in.n_expert_used     = 10;
    in.host_expert_bytes = 410ull << 20;
    in.n_host_experts    = 212;
    in.K                 = 2560;
    in.N                 = 640;
    in.window_us         = 3000.0;
    in.ring_bytes_free   = 820ull << 20;
    in.triplet_bytes_max = 1741 * 1024;
    return in;
}

static moe_exec_layer_inputs gptoss_layer(uint32_t T, bool host) {
    moe_exec_layer_inputs in;
    in.n_tokens          = T;
    in.n_expert          = 32;
    in.n_expert_used     = 4;
    in.host_expert_bytes = host ? 19ull * 37500000ull : 0;
    in.n_host_experts    = host ? 19 : 0;
    in.K                 = 2880;
    in.N                 = 2880;
    in.window_us         = 2000.0;
    in.ring_bytes_free   = 1500ull << 20;
    in.triplet_bytes_max = 37500000ull;
    return in;
}

int main() {
    // p_touch
    CHECK(moe_exec_p_touch(10, 512, 0) == 0.0, "T=0 touches nothing");
    CHECK(std::fabs(moe_exec_p_touch(10, 512, 256) - 0.993) < 1e-2, "T=256 touches 99.3 % (brief section 5)");
    CHECK(std::fabs(moe_exec_p_touch(10, 512, 16) - 0.27) < 1e-2, "T=16 touches 27 %");

    // Qwen3.8 on the B70
    {
        moe_exec_decision d = moe_exec_weigh(qwen_layer(256), b70_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::RING, "Qwen/B70 T=256 answers RING");
        CHECK(d.regime == moe_exec_regime::STREAM_ALL, "Qwen/B70 T=256 streams every host expert");
        CHECK(d.reason == moe_exec_reason::CHEAPER, "chosen on cost");
        CHECK(d.t_ring_us < d.t_cpu_us, "ring is cheaper at T=256");
    }
    {
        moe_exec_decision d = moe_exec_weigh(qwen_layer(4), b70_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::CPU, "Qwen/B70 T=4 answers CPU");
        CHECK(d.regime == moe_exec_regime::ROUTED_ONLY, "T=4 is routed-only");
        moe_exec_layer_inputs wide = qwen_layer(4);
        wide.window_us             = 1e9;
        d                          = moe_exec_weigh(wide, b70_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::CPU, "a routed-only copy is exposed: no window hides it");
    }
    {
        moe_exec_decision d = moe_exec_weigh(qwen_layer(1), b70_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::CPU && d.reason == moe_exec_reason::DECODE, "decode is untouched");
    }
    {
        moe_exec_layer_inputs in = qwen_layer(256);
        in.ring_bytes_free       = 0;
        moe_exec_decision d      = moe_exec_weigh(in, b70_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::CPU && d.reason == moe_exec_reason::RING_UNAVAILABLE, "no ring: CPU");
        d = moe_exec_weigh(in, b70_rates(), moe_exec_policy::RING);
        CHECK(d.choice == moe_exec_choice::CPU, "RING policy cannot force a ring that does not exist");
    }
    {
        moe_exec_decision d = moe_exec_weigh(qwen_layer(256), b70_rates(), moe_exec_policy::CPU);
        CHECK(d.choice == moe_exec_choice::CPU && d.reason == moe_exec_reason::FORCED, "CPU policy forces CPU");
    }
    // GPT-OSS 20B on the B50
    {
        moe_exec_decision d = moe_exec_weigh(gptoss_layer(512, true), b50_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::RING, "GPT-OSS/B50 default-context host layer at T=512 answers RING");
        d = moe_exec_weigh(gptoss_layer(512, false), b50_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::NOT_CONSULTED, "all-VRAM -c 4096 layer is not consulted");
    }
    std::printf("test-moe-exec-cost: %d cases, %d failures\n", g_cases, g_failures);
    return g_failures == 0 ? 0 : 1;
}
