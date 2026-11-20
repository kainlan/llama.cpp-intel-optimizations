// The per-(block, expert) routed-expert counters of the MoE hybrid decode and prompt routes
// (ggml-sycl/moe-route-counts.hpp). SYCL-free: no device is opened and no model is loaded.
#include "ggml-sycl/moe-route-counts.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using ggml_sycl::moe_route_counts;

static int g_failed = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                \
        }                                                              \
    } while (0)

static bool near(double a, double b) {
    return a > b - 1e-9 && a < b + 1e-9;
}

static size_t count_lines(const std::string & s) {
    size_t n = 0;
    for (char c : s) {
        n += c == '\n' ? 1 : 0;
    }
    return n;
}

// Block `b` of `c` gets the decode distribution counts[e] (expert e routed counts[e] times). The experts below
// `n_device` are on the device.
static void route_block(moe_route_counts & c, int b, const std::vector<int> & counts, int n_device) {
    for (size_t e = 0; e < counts.size(); ++e) {
        for (int i = 0; i < counts[e]; ++i) {
            CHECK(c.add_decode(b, static_cast<int>(e), static_cast<int>(e) < n_device));
        }
    }
}

// The same for the prompt counts of block `b`.
static void prompt_block(moe_route_counts & c, int b, const std::vector<int> & counts) {
    for (size_t e = 0; e < counts.size(); ++e) {
        for (int i = 0; i < counts[e]; ++i) {
            CHECK(c.add_prompt(b, static_cast<int>(e)));
        }
    }
}

int main() {
    // 1. The table is sized from the model's dimensions and keeps them exactly, above the 128 x 256 the backend's
    //    layer counters clamp to. Every cell starts at zero, no expert has been seen, no residency is known.
    {
        moe_route_counts c(300, 600);
        CHECK(c.n_blocks() == 300 && c.n_experts() == 600);
        CHECK(c.decode_count(299, 599) == 0 && c.prompt_count(299, 599) == 0);
        CHECK(c.resident_at_last_use(299, 599) == -1);
        CHECK(c.block_resident(299) == -1);
        CHECK(c.add_decode(299, 599, true));
        CHECK(c.add_prompt(299, 599));
        CHECK(c.decode_count(299, 599) == 1 && c.prompt_count(299, 599) == 1);
        CHECK(c.decode_count(299, 598) == 0 && c.decode_count(298, 599) == 0);
        CHECK(c.block_hits(299) == 1 && c.block_misses(299) == 0);
    }
    // 2. Out of range is refused, counted, and writes nothing: a stale table never takes a count from another
    //    model's dimensions.
    {
        moe_route_counts c(48, 512);
        CHECK(!c.add_decode(48, 0, true));
        CHECK(!c.add_decode(0, 512, true));
        CHECK(!c.add_decode(-1, 0, false));
        CHECK(!c.add_decode(0, -1, false));
        CHECK(!c.add_prompt(48, 0));
        CHECK(!c.add_prompt(0, 512));
        CHECK(c.out_of_range() == 6);
        CHECK(ggml_sycl::moe_route_decode_pairs(c) == 0);
        CHECK(c.add_decode(47, 511, false));
        CHECK(c.out_of_range() == 6);
        CHECK(c.n_blocks() == 48 && c.n_experts() == 512);
        moe_route_counts empty(0, 512);
        CHECK(!empty.add_decode(0, 0, true) && !empty.add_prompt(0, 0));
        CHECK(ggml_sycl::moe_route_csv(empty).find('\n') != std::string::npos);
    }
    // 3. Increments: decode and prompt are separate, the hit and miss counters follow the residency each decode
    //    increment carries, and the last residency seen is kept per expert.
    {
        moe_route_counts c(2, 4);
        CHECK(c.add_decode(1, 2, true));
        CHECK(c.add_decode(1, 2, true));
        CHECK(c.add_decode(1, 3, false));
        CHECK(c.add_prompt(1, 2));
        CHECK(c.add_prompt(1, 2));
        CHECK(c.add_prompt(1, 2));
        CHECK(c.decode_count(1, 2) == 2 && c.decode_count(1, 3) == 1);
        CHECK(c.prompt_count(1, 2) == 3 && c.prompt_count(1, 3) == 0);
        CHECK(c.block_hits(1) == 2 && c.block_misses(1) == 1);
        CHECK(c.block_hits(0) == 0 && c.block_misses(0) == 0);
        CHECK(c.resident_at_last_use(1, 2) == 1 && c.resident_at_last_use(1, 3) == 0);
        CHECK(c.resident_at_last_use(1, 0) == -1);
        CHECK(c.add_decode(1, 3, true));
        CHECK(c.resident_at_last_use(1, 3) == 1);
        c.set_block_resident(1, 3);
        CHECK(c.block_resident(1) == 3);
        c.set_block_resident(5, 1);  // out of range: ignored
        CHECK(c.block_resident(5) == -1);
    }
    // 4. Per-block summary and top-N coverage on a hand-made distribution: eight experts, 100 routed pairs, the
    //    first two on the device.
    {
        moe_route_counts c(2, 8);
        route_block(c, 0, { 50, 30, 10, 5, 3, 1, 1, 0 }, 2);
        CHECK(ggml_sycl::moe_route_block_pairs(c, 0) == 100);
        CHECK(c.block_hits(0) == 80 && c.block_misses(0) == 20);
        CHECK(near(ggml_sycl::moe_route_block_hit_rate(c, 0), 0.8));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 0), 0.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 1), 0.5));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 2), 0.8));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 3), 0.9));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 7), 1.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 8), 1.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 100), 1.0));
        // A block nothing was routed in: no pairs, no rate, no coverage.
        CHECK(ggml_sycl::moe_route_block_pairs(c, 1) == 0);
        CHECK(near(ggml_sycl::moe_route_block_hit_rate(c, 1), 0.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 1, 4), 0.0));
        CHECK(ggml_sycl::moe_route_blocks_counted(c) == 1);
    }
    // 4b. Prompt-ranked coverage: the experts are ranked by their PROMPT counts, ties to the lower expert index, and
    //     the figure is the share of the block's decode pairs those experts took.
    {
        moe_route_counts       c(4, 8);
        const std::vector<int> decode0 = { 50, 30, 10, 5, 3, 1, 1, 0 };  // 100 pairs
        const std::vector<int> decode1 = { 0, 0, 0, 0, 10, 10, 0, 0 };   // 20 pairs
        route_block(c, 0, decode0, 2);
        route_block(c, 1, decode1, 0);
        // Block 0: the prompt ranks the experts the way decode does, so it equals the decode top-N coverage.
        prompt_block(c, 0, { 9, 7, 5, 3, 2, 1, 1, 0 });
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 0, 0), 0.0));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 0, 1), 0.5));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 0, 2), 0.8));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 0, 5), 0.98));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 0, 8), 1.0));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 0, 100), 1.0));
        // Block 1: the prompt favours experts 0, 1 and 2 and decode used 4 and 5, so the sets are disjoint until the
        // zero-count ties, broken by lower index, bring in expert 3, then 4 and 5.
        prompt_block(c, 1, { 3, 2, 1, 0, 0, 0, 0, 0 });
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 1, 3), 0.0));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 1, 4), 0.0));  // + expert 3 (tie at 0, lower index)
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 1, 5), 0.5));  // + expert 4
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 1, 6), 1.0));  // + expert 5
        // Block 2: prompt counts only, no decode pairs: 0, and left out of the overall figure.
        prompt_block(c, 2, { 4, 4, 4, 4, 4, 4, 4, 4 });
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(c, 2, 3), 0.0));
        // Overall: (98 + 10) / (100 + 20) at n = 5, (80 + 0) / 120 at n = 2.
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage_overall(c, 5), 108.0 / 120.0));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage_overall(c, 2), 80.0 / 120.0));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage_overall(c, 0), 0.0));
        // Nothing counted: 0.
        moe_route_counts empty(2, 8);
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage_overall(empty, 4), 0.0));
        // A tie: every expert has the same prompt count, so the first n by index are taken, not the hottest in decode.
        moe_route_counts t(1, 8);
        route_block(t, 0, decode0, 2);
        prompt_block(t, 0, { 1, 1, 1, 1, 1, 1, 1, 1 });
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(t, 0, 2), 0.8));
        CHECK(near(ggml_sycl::moe_route_prompt_ranked_coverage(t, 0, 3), 0.9));
        // The reverse tie order would take experts 7 and 6 first: that gives 1 / 100 at n = 2, which it is not.
        CHECK(!near(ggml_sycl::moe_route_prompt_ranked_coverage(t, 0, 2), 0.01));
    }
    // 5. The top experts are the most-routed, not the lowest ids, and ties do not change a sum.
    {
        moe_route_counts c(1, 6);
        route_block(c, 0, { 1, 1, 1, 9, 1, 7 }, 0);  // experts 3 and 5 are the hot ones
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 1), 9.0 / 20.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 2), 16.0 / 20.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage(c, 0, 3), 17.0 / 20.0));
        CHECK(near(ggml_sycl::moe_route_block_hit_rate(c, 0), 0.0));  // all on the host
    }
    // 6. Overall figures over several blocks: the hit rate and the coverage weigh each block by its routed pairs.
    {
        moe_route_counts c(4, 8);
        route_block(c, 0, { 50, 30, 10, 5, 3, 1, 1, 0 }, 2);  // 100 pairs, 80 hits, top-2 = 80
        route_block(c, 2, { 0, 0, 0, 0, 10, 10, 0, 0 }, 0);   // 20 pairs, 0 hits, top-2 = 20
        // Blocks 1 and 3 are never counted.
        CHECK(ggml_sycl::moe_route_blocks_counted(c) == 2);
        CHECK(ggml_sycl::moe_route_decode_pairs(c) == 120);
        CHECK(ggml_sycl::moe_route_decode_hits(c) == 80);
        CHECK(ggml_sycl::moe_route_decode_misses(c) == 40);
        CHECK(near(ggml_sycl::moe_route_overall_hit_rate(c), 80.0 / 120.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage_overall(c, 2), 100.0 / 120.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage_overall(c, 1), 60.0 / 120.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage_overall(c, 8), 1.0));
        moe_route_counts none(4, 8);
        CHECK(near(ggml_sycl::moe_route_overall_hit_rate(none), 0.0));
        CHECK(near(ggml_sycl::moe_route_top_n_coverage_overall(none, 2), 0.0));
        // The measured figure covers the counted blocks only: 80 / 120 above. The model's figure takes the 2 blocks
        // not counted as routing 60 pairs each (120 / 2), all device hits: (80 + 120) / (120 + 120).
        CHECK(near(ggml_sycl::moe_route_model_hit_rate(c), 200.0 / 240.0));
    }
    // 6b. The model hit rate by hand: 2 counted blocks with 100 pairs and 10 hits between them, 1 block not counted,
    //     which routed 100 / 2 = 50 pairs, all hits: (10 + 50) / (100 + 50) = 40 %.
    {
        moe_route_counts c(3, 8);
        route_block(c, 0, { 10, 50, 0, 0, 0, 0, 0, 0 }, 1);  // 60 pairs, 10 hits
        route_block(c, 1, { 0, 40, 0, 0, 0, 0, 0, 0 }, 1);   // 40 pairs, 0 hits
        CHECK(ggml_sycl::moe_route_decode_pairs(c) == 100 && ggml_sycl::moe_route_decode_hits(c) == 10);
        CHECK(near(ggml_sycl::moe_route_overall_hit_rate(c), 0.1));
        CHECK(near(ggml_sycl::moe_route_model_hit_rate(c), 0.4));
        // Every block counted: nothing to take as all device, so the two figures agree.
        route_block(c, 2, { 5, 5, 0, 0, 0, 0, 0, 0 }, 1);  // 10 pairs, 5 hits
        CHECK(near(ggml_sycl::moe_route_model_hit_rate(c), ggml_sycl::moe_route_overall_hit_rate(c)));
        CHECK(near(ggml_sycl::moe_route_model_hit_rate(c), 15.0 / 110.0));
        // Nothing counted: no average to take from, so 0 and not a division by zero.
        moe_route_counts none(3, 8);
        CHECK(near(ggml_sycl::moe_route_model_hit_rate(none), 0.0));
        moe_route_counts empty(0, 8);
        CHECK(near(ggml_sycl::moe_route_model_hit_rate(empty), 0.0));
    }
    // 7. Mean resident experts per block: a counted block contributes what the plan holds on the device, a block
    //    that was not counted is all resident (the direct route reports nothing), a counted block with no recorded
    //    residency is left out.
    {
        moe_route_counts c(4, 10);
        route_block(c, 0, { 1 }, 1);
        route_block(c, 1, { 1 }, 0);
        route_block(c, 2, { 1 }, 0);
        c.set_block_resident(0, 4);
        c.set_block_resident(1, 0);
        // block 2: counted, residency unknown; block 3: not counted
        CHECK(near(ggml_sycl::moe_route_mean_resident_experts(c), (4.0 + 0.0 + 10.0) / 3.0));
    }
    // 8. The model's own dimensions, 48 blocks of 512 experts: every cell is written, the CSV has one row per
    //    (block, expert) and a header, and a cell reads back where it was written.
    {
        moe_route_counts c(48, 512);
        for (int b = 0; b < 48; ++b) {
            c.set_block_resident(b, b < 13 ? 512 : 0);
        }
        for (int b = 13; b < 48; ++b) {
            for (int i = 0; i < 10; ++i) {
                CHECK(c.add_decode(b, (b * 7 + i) % 512, false));
            }
        }
        CHECK(c.add_prompt(0, 511));
        CHECK(ggml_sycl::moe_route_blocks_counted(c) == 35);
        CHECK(ggml_sycl::moe_route_decode_pairs(c) == 350);
        CHECK(near(ggml_sycl::moe_route_overall_hit_rate(c), 0.0));
        const std::string csv = ggml_sycl::moe_route_csv(c);
        CHECK(count_lines(csv) == 48 * 512 + 1);
        CHECK(csv.rfind("block,expert,decode_count,prompt_count,resident_at_last_use\n0,0,", 0) == 0);
        CHECK(csv.find("\n0,511,0,1,-1\n") != std::string::npos);
        CHECK(csv.find("\n13,91,1,0,0\n") != std::string::npos);  // 13 * 7 % 512 = 91, routed once on the host
        CHECK(csv.find("\n47,0,0,0,-1\n") != std::string::npos);
    }
    // 9. CSV text of a small table: block-major rows, decode then prompt then residency.
    {
        moe_route_counts c(2, 2);
        CHECK(c.add_decode(0, 1, true));
        CHECK(c.add_decode(0, 1, true));
        CHECK(c.add_prompt(0, 1));
        CHECK(c.add_decode(1, 0, false));
        CHECK(c.add_prompt(1, 1));
        const std::string expect =
            "block,expert,decode_count,prompt_count,resident_at_last_use\n"
            "0,0,0,0,-1\n"
            "0,1,2,1,1\n"
            "1,0,1,0,0\n"
            "1,1,0,1,-1\n";
        CHECK(ggml_sycl::moe_route_csv(c) == expect);
    }
    // 10. The one WARN line: tokens, blocks counted and not counted, routed pairs, the decode hit rate, and the top-N
    //     coverage at the mean number of resident experts per block.
    {
        moe_route_counts c(4, 8);
        route_block(c, 0, { 50, 30, 10, 5, 3, 1, 1, 0 }, 2);
        route_block(c, 2, { 0, 0, 0, 0, 10, 10, 0, 0 }, 0);
        prompt_block(c, 0, { 9, 7, 5, 3, 2, 1, 1, 0 });  // ranks like decode: its top 5 take 98 of 100
        prompt_block(c, 2, { 3, 2, 1, 0, 0, 0, 0, 0 });  // ranks experts 0, 1, 2, 3, 4: takes only 10 of 20
        c.set_block_resident(0, 2);
        c.set_block_resident(2, 0);
        // mean resident = (2 + 0 + 8 + 8) / 4 = 4.5, which the line rounds to 5 slots per block
        const std::string line = ggml_sycl::moe_route_summary_line(c, 7);
        CHECK(line.find('\n') == std::string::npos);
        CHECK(line.find("[MOE-ROUTE-COUNTS]") == 0);
        CHECK(line.find("process tokens=7") != std::string::npos);
        CHECK(line.find("blocks counted=2") != std::string::npos);
        CHECK(line.find("not counted=2 (all resident, not counted)") != std::string::npos);
        CHECK(line.find("of 4") != std::string::npos);
        CHECK(line.find("decode routed pairs=120") != std::string::npos);
        CHECK(
            line.find(
                "hit rate=66.7% (device 80, host 40) model hit rate=83.3% (blocks not counted taken as all device);") !=
            std::string::npos);
        CHECK(line.find("top-5 coverage=") != std::string::npos);
        // top-5 of block 0 = 98, of block 2 = 20: 118 / 120 = 98.3%
        CHECK(line.find("top-5 coverage=98.3%") != std::string::npos);
        CHECK(line.find("over the counted blocks (each given the model's mean 4.5 slots);") != std::string::npos);
        // prompt-ranked top-5: block 0 takes 98, block 2 takes 10: 108 / 120 = 90.0%
        CHECK(line.find("; prompt-ranked top-5 coverage=90.0%; ") != std::string::npos);
        CHECK(line.find("prompt routed pairs=34") != std::string::npos);
    }
    // 11. The per-block table: one line per block, naming a block that was not counted as all resident.
    {
        moe_route_counts c(3, 8);
        route_block(c, 1, { 5, 5, 0, 0, 0, 0, 0, 0 }, 1);
        CHECK(c.add_prompt(0, 3));
        const std::string table = ggml_sycl::moe_route_block_table(c);
        CHECK(count_lines(table) == 3);
        CHECK(table.find("block 1 ") != std::string::npos && table.find("hit rate=50.0%") != std::string::npos);
        CHECK(table.find("block 0 all resident, not counted") != std::string::npos);
        CHECK(table.find("block 2 all resident, not counted") != std::string::npos);
    }
    // 12. reset() zeroes every counter and the residency, and forgets the last reported total.
    {
        moe_route_counts c(2, 4);
        CHECK(c.add_decode(1, 1, true));
        CHECK(c.add_prompt(1, 1));
        c.set_block_resident(1, 2);
        CHECK(!c.add_decode(2, 0, true));
        CHECK(c.claim_report(1));
        CHECK(!c.claim_report(1));
        c.reset();
        CHECK(c.decode_count(1, 1) == 0 && c.prompt_count(1, 1) == 0);
        CHECK(c.block_hits(1) == 0 && c.block_misses(1) == 0);
        CHECK(c.resident_at_last_use(1, 1) == -1 && c.block_resident(1) == -1);
        CHECK(c.out_of_range() == 0);
        CHECK(c.claim_report(1));
        CHECK(c.n_blocks() == 2 && c.n_experts() == 4);
    }
    // 12b. A report is claimed when the decode pairs grew since the last one claimed: nothing counted claims nothing,
    //      the same total claims once, a larger total claims again, and a smaller one never does.
    {
        moe_route_counts c(2, 4);
        CHECK(!c.claim_report(0));
        CHECK(c.claim_report(3));
        CHECK(!c.claim_report(3));
        CHECK(!c.claim_report(2));
        CHECK(c.claim_report(4));
        CHECK(!c.claim_report(4));
        CHECK(c.claim_report(1000));
        // Several threads that count the same total claim it once between them.
        moe_route_counts         racy(1, 1);
        std::atomic<int>         claims{ 0 };
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&racy, &claims]() {
                for (uint64_t total = 1; total <= 50; ++total) {
                    claims.fetch_add(racy.claim_report(total) ? 1 : 0);
                }
            });
        }
        for (std::thread & th : threads) {
            th.join();
        }
        CHECK(claims.load() >= 1 && claims.load() <= 50);
        CHECK(!racy.claim_report(50));
        CHECK(racy.claim_report(51));
    }
    // 13. Increments from several threads at once do not lose a count (the pool worker and the prompt thread).
    {
        moe_route_counts         c(2, 4);
        const int                n_threads = 4;
        const int                per       = 20000;
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&c, t]() {
                for (int i = 0; i < per; ++i) {
                    c.add_decode(1, 2, (t & 1) != 0);
                    c.add_prompt(1, 2);
                }
            });
        }
        for (std::thread & th : threads) {
            th.join();
        }
        CHECK(c.decode_count(1, 2) == static_cast<uint64_t>(n_threads) * per);
        CHECK(c.prompt_count(1, 2) == static_cast<uint64_t>(n_threads) * per);
        CHECK(c.block_hits(1) == static_cast<uint64_t>(n_threads / 2) * per);
        CHECK(c.block_misses(1) == static_cast<uint64_t>(n_threads / 2) * per);
    }

    if (g_failed != 0) {
        std::printf("test-sycl-moe-route-counts: %d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test-sycl-moe-route-counts: all checks passed\n");
    return 0;
}
