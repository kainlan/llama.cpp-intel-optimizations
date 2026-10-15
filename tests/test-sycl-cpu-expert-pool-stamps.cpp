// Per-job critical-path stamps of the CPU expert pool trace (llama.cpp-j6jf):
// the per-token sums the [CPU-EXPERT-POOL-TRACE] line carries, the clamp
// rules, and the per-job line's fields.
//
// ggml-sycl/cpu-expert-pool-stamps.hpp includes only the standard library, so
// this test links nothing and opens no device.

#include "ggml-sycl/cpu-expert-pool-stamps.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using ggml_sycl::cpu_expert_job_kind;
using ggml_sycl::cpu_expert_job_stamp_sums;
using ggml_sycl::cpu_expert_job_stamps;

static int g_failures = 0;

static void expect_near(const char * what, double got, double want) {
    if (std::fabs(got - want) > 1e-6) {
        std::printf("FAIL: %s = %.3f, want %.3f\n", what, got, want);
        g_failures++;
    }
}

static void expect_eq(const char * what, unsigned long long got, unsigned long long want) {
    if (got != want) {
        std::printf("FAIL: %s = %llu, want %llu\n", what, got, want);
        g_failures++;
    }
}

static void expect_true(const char * what, bool ok) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        g_failures++;
    }
}

// Times in us, stored as ns like the trace stores them.
static cpu_expert_job_stamps
job(double submit, double inputs, double cstart, double cend, double jentry, double jexit, bool ready) {
    cpu_expert_job_stamps s;
    s.t_submit_ns        = static_cast<int64_t>(submit * 1000.0);
    s.t_inputs_ready_ns  = static_cast<int64_t>(inputs * 1000.0);
    s.t_compute_start_ns = static_cast<int64_t>(cstart * 1000.0);
    s.t_compute_end_ns   = static_cast<int64_t>(cend * 1000.0);
    s.t_join_entry_ns    = jentry < 0 ? 0 : static_cast<int64_t>(jentry * 1000.0);
    s.t_join_exit_ns     = jexit < 0 ? 0 : static_cast<int64_t>(jexit * 1000.0);
    s.join_found_ready   = ready;
    return s;
}

static void test_sums() {
    // Base 1000 us so no stamp is 0 (0 means "not stamped").
    std::vector<cpu_expert_job_stamps> jobs;
    // Job A: joined after it finished.
    jobs.push_back(job(1000, 1100, 1150, 1400, 1450, 1460, true));
    // Job B: inputs ready 40 us after A's join returned; its join waited for it.
    jobs.push_back(job(1010, 1500, 1520, 1800, 1700, 1810, false));
    // Job C: inputs ready before B's join returned (gap clamps to 0); never joined.
    jobs.push_back(job(1020, 1805, 1815, 1900, -1, -1, false));
    // Job D: its predecessor C was never joined, so no gap is counted; joined ready.
    jobs.push_back(job(1030, 2000, 2010, 2100, 2130, 2135, true));

    // The order of the input does not matter: the sums take submission order.
    std::vector<cpu_expert_job_stamps> shuffled = { jobs[2], jobs[0], jobs[3], jobs[1] };
    const cpu_expert_job_stamp_sums    s        = ggml_sycl::cpu_expert_job_stamps_sum(shuffled);

    expect_eq("jobs", s.jobs, 4);
    expect_eq("joined", s.joined, 3);
    expect_eq("joins_waited", s.joins_waited, 1);
    // inputs_wait: (1100-1000) + (1500-1010) + (1805-1020) + (2000-1030) = 100 + 490 + 785 + 970
    expect_near("inputs_wait_us", s.inputs_wait_us, 2345.0);
    // chain_gap: B after A: 1500-1460 = 40; C after B: 1805-1810 < 0 -> 0; D after C: C unjoined -> none
    expect_near("chain_gap_us", s.chain_gap_us, 40.0);
    // join_lat over ready joins: A 1450-1400 = 50, D 2130-2100 = 30
    expect_near("join_lat_us", s.join_lat_us, 80.0);
    // join_waited over joins that waited: B 1810-1700 = 110
    expect_near("join_waited_us", s.join_waited_us, 110.0);
    // pre_compute: 50 + 20 + 10 + 10
    expect_near("pre_compute_us", s.pre_compute_us, 90.0);
}

static void test_clamps() {
    // A ready join whose entry stamp precedes the compute end (clock skew between
    // threads) adds 0 latency, not a negative one.
    std::vector<cpu_expert_job_stamps> jobs;
    jobs.push_back(job(1000, 1010, 1020, 1100, 1090, 1095, true));
    // compute start before inputs ready cannot happen, but clamps to 0 if it does.
    jobs.push_back(job(1001, 1200, 1190, 1300, 1310, 1320, true));
    const cpu_expert_job_stamp_sums s = ggml_sycl::cpu_expert_job_stamps_sum(jobs);
    expect_near("clamped join_lat_us", s.join_lat_us, 10.0);
    expect_near("clamped pre_compute_us", s.pre_compute_us, 10.0);
    expect_near("chain_gap after a ready join", s.chain_gap_us, 1200.0 - 1095.0);

    // A job with no inputs-ready stamp contributes to no sum but is counted.
    std::vector<cpu_expert_job_stamps> partial(1);
    partial[0].t_submit_ns            = 5000;
    const cpu_expert_job_stamp_sums p = ggml_sycl::cpu_expert_job_stamps_sum(partial);
    expect_eq("partial jobs", p.jobs, 1);
    expect_near("partial inputs_wait_us", p.inputs_wait_us, 0.0);
    expect_near("partial pre_compute_us", p.pre_compute_us, 0.0);

    const cpu_expert_job_stamp_sums e = ggml_sycl::cpu_expert_job_stamps_sum({});
    expect_eq("empty jobs", e.jobs, 0);
    expect_near("empty chain_gap_us", e.chain_gap_us, 0.0);
}

static void test_kind() {
    expect_true("down", ggml_sycl::cpu_expert_job_kind_from_name("blk.7.ffn_down_exps.weight") ==
                            ggml_sycl::CPU_EXPERT_JOB_KIND_DOWN);
    expect_true("gate", ggml_sycl::cpu_expert_job_kind_from_name("blk.7.ffn_gate_exps.weight") ==
                            ggml_sycl::CPU_EXPERT_JOB_KIND_GATE);
    expect_true("up", ggml_sycl::cpu_expert_job_kind_from_name("blk.7.ffn_up_exps.weight") ==
                          ggml_sycl::CPU_EXPERT_JOB_KIND_UP);
    expect_true("gate_up", ggml_sycl::cpu_expert_job_kind_from_name("blk.7.ffn_gate_up_exps.weight") ==
                               ggml_sycl::CPU_EXPERT_JOB_KIND_GATE_UP);
    expect_true("other",
                ggml_sycl::cpu_expert_job_kind_from_name("output.weight") == ggml_sycl::CPU_EXPERT_JOB_KIND_OTHER);
    expect_true("null", ggml_sycl::cpu_expert_job_kind_from_name(nullptr) == ggml_sycl::CPU_EXPERT_JOB_KIND_OTHER);
    expect_true("kind name",
                std::strcmp(ggml_sycl::cpu_expert_job_kind_name(ggml_sycl::CPU_EXPERT_JOB_KIND_DOWN), "down") == 0);
}

static void test_format() {
    cpu_expert_job_stamps s = job(1010, 1100, 1150, 1400, 1450, 1460, true);
    s.layer                 = 12;
    s.kind                  = ggml_sycl::CPU_EXPERT_JOB_KIND_DOWN;
    s.rows                  = 640;
    s.bytes                 = 123456;
    char buf[512];
    ggml_sycl::cpu_expert_job_stamps_format(buf, sizeof(buf), 3, s, 1000 * 1000);
    const std::string want =
        "job=3 layer=12 kind=down rows=640 bytes=123456 submit=10us inputs_ready=100us compute_start=150us "
        "compute_end=400us join_entry=450us join_exit=460us join=ready";
    if (want != buf) {
        std::printf("FAIL: per-job line\n  got:  %s\n  want: %s\n", buf, want.c_str());
        g_failures++;
    }

    cpu_expert_job_stamps u = job(1000, 1000, 1000, 1000, -1, -1, false);
    ggml_sycl::cpu_expert_job_stamps_format(buf, sizeof(buf), 0, u, 1000 * 1000);
    expect_true("unjoined job prints join=none", std::strstr(buf, "join_entry=- join_exit=- join=none") != nullptr);

    cpu_expert_job_stamps w = job(1000, 1000, 1000, 1000, 1001, 1002, false);
    ggml_sycl::cpu_expert_job_stamps_format(buf, sizeof(buf), 0, w, 1000 * 1000);
    expect_true("waited join prints join=waited", std::strstr(buf, "join=waited") != nullptr);
}

int main() {
    test_sums();
    test_clamps();
    test_kind();
    test_format();
    if (g_failures != 0) {
        std::printf("test-sycl-cpu-expert-pool-stamps: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-cpu-expert-pool-stamps: PASS\n");
    return 0;
}
