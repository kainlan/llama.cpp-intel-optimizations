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
    r.h2d_gbps     = 26.5;     // measured, 8bgi c-nn1q
    r.cpu_gbps     = 22.0;     // measured, 7fmt table (c-plz8 / c-ncj4)
    r.reorder_gbps = 200.0;    // seed (Task 9 / probes replace it)
    r.dev_gflops   = 50000.0;  // seed (Task 9 / probes replace it)
    r.cpu_gflops   = 1000.0;   // seed (Task 9 / probes replace it)
    r.cpu_fixed_us = 500.0;    // seed (Task 9 / probes replace it)
    r.dev_fixed_us = 800.0;    // seed (Task 9 / probes replace it)
    return r;
}

static moe_exec_rates b50_rates() {
    moe_exec_rates r = b70_rates();
    r.h2d_gbps       = 27.1;     // measured, 8bgi c-nn1q
    r.cpu_gbps       = 5.6;      // measured, 7fmt table (c-plz8 / c-ncj4)
    r.dev_gflops     = 25000.0;  // seed (Task 9 / probes replace it)
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
    in.triplet_bytes_max = 2048 * 1024;  // largest host triplet, rounded up from the 410 MiB / 212 mean
    return in;
}

static moe_exec_layer_inputs gptoss_layer(uint32_t T, bool host) {
    moe_exec_layer_inputs in;
    in.n_tokens      = T;
    in.n_expert      = 32;
    in.n_expert_used = 4;
    in.host_expert_bytes =
        host ? 19ull * 13200000ull : 0;  // 13.2 MB per expert triplet: 10152345600 B / (24 x 32), test-kv-room-cost
    in.n_host_experts    = host ? 19 : 0;
    in.K                 = 2880;
    in.N                 = 2880;
    in.window_us         = 2000.0;
    in.ring_bytes_free   = 1500ull << 20;
    in.triplet_bytes_max = 13200000ull;
    return in;
}

int main() {
    // p_touch
    CHECK(moe_exec_p_touch(10, 512, 0) == 0.0, "T=0 touches nothing");
    CHECK(std::fabs(moe_exec_p_touch(10, 512, 256) - 0.993) < 1e-3, "T=256 touches 99.3 % (brief section 5)");
    CHECK(std::fabs(moe_exec_p_touch(10, 512, 16) - 0.27) < 1e-2, "T=16 touches 27 %");

    // Qwen3.8 on the B70
    {
        moe_exec_decision d = moe_exec_weigh(qwen_layer(256), b70_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::RING, "Qwen/B70 T=256 answers RING");
        CHECK(d.regime == moe_exec_regime::STREAM_ALL, "Qwen/B70 T=256 streams every host expert");
        CHECK(d.reason == moe_exec_reason::CHEAPER, "chosen on cost");
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
        // STREAM_ALL earns the window: with a fixed 2000 us device cost the window decides the verdict.
        moe_exec_rates r         = b70_rates();
        r.dev_fixed_us           = 2000.0;
        moe_exec_layer_inputs in = qwen_layer(256);
        moe_exec_decision     d  = moe_exec_weigh(in, r, moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::RING, "window 3000 us hides the stream-all copy: RING");
        in.window_us = 0.0;
        d            = moe_exec_weigh(in, r, moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::CPU, "without the window the same layer is CPU");
    }
    {
        // stream_all_p_min boundary: p exactly at the threshold streams all, just above it is routed-only.
        moe_exec_rates r   = b70_rates();
        const double   p   = moe_exec_p_touch(10, 512, 256);
        r.stream_all_p_min = p;
        CHECK(moe_exec_weigh(qwen_layer(256), r, moe_exec_policy::AUTO).regime == moe_exec_regime::STREAM_ALL,
              "p at the threshold is STREAM_ALL");
        r.stream_all_p_min = p + 1e-9;
        CHECK(moe_exec_weigh(qwen_layer(256), r, moe_exec_policy::AUTO).regime == moe_exec_regime::ROUTED_ONLY,
              "p just below the threshold is ROUTED_ONLY");
    }
    {
        // RING policy forces the ring even where the CPU is cheaper.
        moe_exec_decision d = moe_exec_weigh(qwen_layer(4), b70_rates(), moe_exec_policy::RING);
        CHECK(d.choice == moe_exec_choice::RING && d.reason == moe_exec_reason::FORCED, "RING policy forces RING");
        // A ring smaller than the largest triplet cannot stage it.
        moe_exec_layer_inputs in = qwen_layer(256);
        in.ring_bytes_free       = 1024;
        d                        = moe_exec_weigh(in, b70_rates(), moe_exec_policy::AUTO);
        CHECK(d.choice == moe_exec_choice::CPU && d.reason == moe_exec_reason::RING_UNAVAILABLE,
              "ring smaller than the largest triplet is unavailable");
    }
    {
        // Compute-bound CPU: a slow CPU GEMM rate dominates its bandwidth term.
        moe_exec_rates r            = b70_rates();
        r.cpu_gflops                = 1.0;
        moe_exec_layer_inputs in    = qwen_layer(256);
        const double          gflop = moe_exec_layer_gflop(in) * (static_cast<double>(in.n_host_experts) / in.n_expert);
        moe_exec_decision     d     = moe_exec_weigh(in, r, moe_exec_policy::AUTO);
        CHECK(std::fabs(d.t_cpu_us - (gflop / r.cpu_gflops * 1e6 + r.cpu_fixed_us)) < 1e-6,
              "compute-bound CPU cost is gflop / rate + fixed");
    }
    {
        CHECK(moe_exec_rates_valid(b70_rates()), "the B70 rates are valid");
        CHECK(!moe_exec_rates_valid(moe_exec_rates{}), "default-constructed (unmeasured) rates are invalid");
        moe_exec_rates r = b70_rates();
        r.cpu_gbps       = 0.0;
        CHECK(!moe_exec_rates_valid(r), "a zero rate is invalid");
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
        CHECK(d.reason == moe_exec_reason::RING_UNAVAILABLE, "forced RING with no ring reports RING_UNAVAILABLE");
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
        CHECK(d.choice == moe_exec_choice::NOT_CONSULTED && d.reason == moe_exec_reason::NO_HOST_EXPERTS,
              "all-VRAM -c 4096 layer is not consulted");
    }
    std::printf("test-moe-exec-cost: %d cases, %d failures\n", g_cases, g_failures);
    return g_failures == 0 ? 0 : 1;
}
