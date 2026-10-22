// test-moe-exec-cost: the MoE executor's per-layer cost (moe-exec-cost.hpp; llama.cpp-36j0) on the measured shapes.
// Host only: no device, no model, no SYCL.
#include "moe-exec-cost.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
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
    // seed: the Qwen GSQ-RCO mix (seven measured types at 22 GB/s) gives 22; a mix with an unmeasured type keeps
    // the measured bytes' mean (the room refuses unmeasured types; the executor just prices them at the default)
    {
        std::vector<std::pair<ggml_type, size_t>> mix = {
            { GGML_TYPE_IQ3_S,  218 },
            { GGML_TYPE_IQ4_NL, 198 },
            { GGML_TYPE_Q2_0,   165 },
            { GGML_TYPE_IQ2_S,  125 }
        };
        moe_exec_rates r = moe_exec_rates_seed(mix);
        CHECK(std::fabs(r.cpu_gbps - 22.0) < 1e-9, "Qwen mix seeds the mix's 22 GB/s");
        CHECK(r.h2d_gbps == 20.0 && r.reorder_gbps == 200.0, "H2D and reorder carry their seeds");
        CHECK(moe_exec_rates_valid(r), "the seed passes the validity check weigh asserts");
        mix.push_back({ GGML_TYPE_Q4_0, 1000 });
        r = moe_exec_rates_seed(mix);
        CHECK(std::fabs(r.cpu_gbps - 22.0) < 1e-9, "an unmeasured type does not move the measured mean");
        r = moe_exec_rates_seed({
            { GGML_TYPE_Q4_0, 1000 }
        });
        CHECK(r.cpu_gbps == PLACEMENT_CPU_EXPERT_GBPS_DEFAULT, "no measured type: the room's default");
        CHECK(moe_exec_rates_valid(r), "the all-default seed is valid too");
        r = moe_exec_rates_seed({
            { GGML_TYPE_MXFP4, 100 },
            { GGML_TYPE_IQ3_S, 100 }
        });
        CHECK(std::fabs(r.cpu_gbps - 200.0 / (100.0 / 5.6 + 100.0 / 22.0)) < 1e-9, "two measured types: harmonic mean");
    }
    // dev_gflops seeds per type from the Task 9 measurement (llama.cpp-g4uy), byte-weighted harmonic mean over the mix:
    // bytes / sum(bytes_i / rate_i). The fixtures above that call weigh set their own rates, so none of their
    // expectations depended on the old 20000 seed; the only pin that did is the unset-variable check below (600).
    {
        moe_exec_rates r = moe_exec_rates_seed({
            { GGML_TYPE_MXFP4, 1000 }
        });
        CHECK(r.dev_gflops == 3990.0, "a pure MXFP4 mix seeds the batched XMX executor's 3990 GFLOP/s");
        r = moe_exec_rates_seed({
            { GGML_TYPE_Q4_0, 1000 }
        });
        CHECK(r.dev_gflops == 630.0, "a pure Q4_0 mix seeds the per-row MMVQ path's 630 GFLOP/s");
        r = moe_exec_rates_seed({
            { GGML_TYPE_IQ3_S, 1000 }
        });
        CHECK(r.dev_gflops == 600.0, "a type with no batched executor seeds the per-row 600 GFLOP/s");
        r = moe_exec_rates_seed({
            { GGML_TYPE_MXFP4, 500 },
            { GGML_TYPE_Q4_0,  500 }
        });
        CHECK(std::fabs(r.dev_gflops - 2.0 / (1.0 / 3990.0 + 1.0 / 630.0)) < 0.1 &&
                  std::fabs(r.dev_gflops - 1088.1) < 0.1,
              "equal MXFP4 and Q4_0 bytes seed the harmonic mean, 1088.1 (not the 2310 arithmetic mean)");
        // unequal bytes tell the byte weighting from an unweighted harmonic mean (which would give the 1088.1 above):
        // 400 / (100 / 3990 + 300 / 630) = 400 / (0.025063 + 0.476190) = 798.0
        r = moe_exec_rates_seed({
            { GGML_TYPE_MXFP4, 100 },
            { GGML_TYPE_Q4_0,  300 }
        });
        CHECK(std::fabs(r.dev_gflops - 400.0 / (100.0 / 3990.0 + 300.0 / 630.0)) < 0.1 &&
                  std::fabs(r.dev_gflops - 798.0) < 0.1,
              "100 B MXFP4 + 300 B Q4_0 seed the byte-weighted harmonic mean, 798.0");
        r = moe_exec_rates_seed({});
        CHECK(r.dev_gflops == 600.0, "an empty mix seeds 600 GFLOP/s");
        CHECK(moe_exec_rates_valid(r), "the empty-mix seed is valid");
        // the env override still wins over a per-type seed
        auto env_fn = [](const char * name) -> const char * {
            return std::string(name) == "GGML_SYCL_MOE_EXEC_DEV_GFLOPS" ? "777" : nullptr;
        };
        r = moe_exec_rates_seed({
            { GGML_TYPE_MXFP4, 1000 }
        });
        moe_exec_rates_apply_env(r, env_fn);
        CHECK(r.dev_gflops == 777.0, "the env override wins over the per-type seed");
    }
    // env overrides follow the w013 rule
    {
        static const char * fake_env_h2d = "31.5";
        auto                getenv_fn    = [](const char * name) -> const char * {
            return std::string(name) == "GGML_SYCL_MOE_EXEC_H2D_GBPS" ? fake_env_h2d : nullptr;
        };
        moe_exec_rates r = moe_exec_rates_seed({});
        moe_exec_rates_apply_env(r, getenv_fn);
        CHECK(r.h2d_gbps == 31.5, "H2D override applied");
        CHECK(r.cpu_gbps == PLACEMENT_CPU_EXPERT_GBPS_DEFAULT && r.reorder_gbps == 200.0 && r.dev_gflops == 600.0,
              "an unset variable keeps its seed (the empty mix's 600 GFLOP/s)");
        fake_env_h2d = "-3";
        r            = moe_exec_rates_seed({});
        moe_exec_rates_apply_env(r, getenv_fn);
        CHECK(r.h2d_gbps == 20.0, "a negative override keeps the seed");
        fake_env_h2d = "nan";
        r            = moe_exec_rates_seed({});
        moe_exec_rates_apply_env(r, getenv_fn);
        CHECK(r.h2d_gbps == 20.0, "a non-finite override keeps the seed");
        fake_env_h2d = "fast";
        r            = moe_exec_rates_seed({});
        moe_exec_rates_apply_env(r, getenv_fn);
        CHECK(r.h2d_gbps == 20.0, "an unparsable override keeps the seed");
        fake_env_h2d = "0";
        r            = moe_exec_rates_seed({});
        moe_exec_rates_apply_env(r, getenv_fn);
        CHECK(r.h2d_gbps == 20.0, "a zero override keeps the seed");
        auto all_fn = [](const char * name) -> const char * {
            const std::string n(name);
            return n == "GGML_SYCL_MOE_EXEC_H2D_GBPS"     ? "1" :
                   n == "GGML_SYCL_MOE_EXEC_CPU_GBPS"     ? "2" :
                   n == "GGML_SYCL_MOE_EXEC_DEV_GFLOPS"   ? "3" :
                   n == "GGML_SYCL_MOE_EXEC_REORDER_GBPS" ? "4" :
                                                            nullptr;
        };
        r = moe_exec_rates_seed({});
        moe_exec_rates_apply_env(r, all_fn);
        CHECK(r.h2d_gbps == 1.0 && r.cpu_gbps == 2.0 && r.dev_gflops == 3.0 && r.reorder_gbps == 4.0,
              "each of the four variables overrides its own rate");
        moe_exec_rates_apply_env(r, nullptr);
        CHECK(r.h2d_gbps == 1.0, "a null getenv keeps what the record holds");
    }
    // the H2D probe's aggregation (llama.cpp-xr4i): the max over iterations 2..n, so one slow window cannot drag the
    // persisted rate down, and none when even that max is under half the seed
    {
        double       lo        = -1.0;
        double       hi        = -1.0;
        const double steady[8] = { 9.3, 25.1, 24.8, 26.0, 25.5, 24.9, 25.2, 25.0 };
        double       r         = moe_exec_h2d_rate_from_iterations(steady, 8, 20.0, &lo, &hi);
        CHECK(r == 26.0 && hi == 26.0, "the rate is the max over iterations 2..8");
        CHECK(lo == 24.8, "the spread's low end is the min over iterations 2..8, not the slow first iteration");
        const double one_slow[8] = { 25.0, 25.3, 9.4, 25.6, 25.1, 25.4, 25.2, 25.5 };
        r                        = moe_exec_h2d_rate_from_iterations(one_slow, 8, 20.0, &lo, &hi);
        CHECK(r == 25.6 && lo == 9.4 && hi == 25.6, "one slow window among 2..8 neither lowers the rate nor hides");
        const double fast_first[8] = { 31.0, 24.0, 25.0, 24.5, 24.0, 24.2, 24.1, 24.4 };
        r                          = moe_exec_h2d_rate_from_iterations(fast_first, 8, 20.0, &lo, &hi);
        CHECK(r == 25.0 && hi == 25.0, "iteration 1 is a warm-up: a fast one is not the rate");
        const double slow[8] = { 9.3, 9.8, 9.6, 9.5, 9.9, 9.7, 9.6, 9.4 };
        r                    = moe_exec_h2d_rate_from_iterations(slow, 8, 20.0, &lo, &hi);
        CHECK(r == 0.0, "a max under half the seed yields no rate, so the probe retries");
        CHECK(lo == 9.4 && hi == 9.9, "the spread is still reported for a rejected window");
        const double edge[3] = { 1.0, 10.0, 9.0 };
        CHECK(moe_exec_h2d_rate_from_iterations(edge, 3, 20.0, &lo, &hi) == 10.0, "exactly half the seed is kept");
        const double under[3] = { 1.0, 9.99, 9.0 };
        CHECK(moe_exec_h2d_rate_from_iterations(under, 3, 20.0, &lo, &hi) == 0.0, "just under half the seed is not");
        const double bad[8] = { 25.0, 0.0, -3.0, NAN, INFINITY, 12.0, 0.0, 14.0 };
        r                   = moe_exec_h2d_rate_from_iterations(bad, 8, 20.0, &lo, &hi);
        CHECK(r == 14.0 && lo == 12.0 && hi == 14.0, "zero, negative and non-finite iterations are skipped");
        const double none[4] = { 25.0, 0.0, NAN, -1.0 };
        r                    = moe_exec_h2d_rate_from_iterations(none, 4, 20.0, &lo, &hi);
        CHECK(r == 0.0 && lo == 0.0 && hi == 0.0, "no valid iteration yields no rate and a zero spread");
        const double two[2] = { 30.0, 22.0 };
        CHECK(moe_exec_h2d_rate_from_iterations(two, 2, 20.0, nullptr, nullptr) == 22.0,
              "two iterations use the second alone, and the spread outputs are optional");
        CHECK(moe_exec_h2d_rate_from_iterations(two, 1, 20.0, &lo, &hi) == 0.0 && lo == 0.0 && hi == 0.0,
              "a single iteration is only the warm-up");
        CHECK(moe_exec_h2d_rate_from_iterations(nullptr, 0, 20.0, &lo, &hi) == 0.0, "no iterations yield no rate");
        const moe_exec_rates seed = moe_exec_rates_seed({});
        CHECK(moe_exec_h2d_rate_from_iterations(slow, 8, seed.h2d_gbps, nullptr, nullptr) == 0.0 &&
                  moe_exec_h2d_rate_from_iterations(steady, 8, seed.h2d_gbps, nullptr, nullptr) == 26.0,
              "the bimodal readings of llama.cpp-zujy c-vbf4: 9.x GB/s rejected, 24-26 GB/s kept, against the seed");
    }
    std::printf("test-moe-exec-cost: %d cases, %d failures\n", g_cases, g_failures);
    return g_failures == 0 ? 0 : 1;
}
