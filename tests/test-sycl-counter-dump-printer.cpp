// The SYCL exit counter dump, and the zone chokepoint that feeds three of its counters.
//
// scripts/check-sycl-counter-dump.py proves the table is compiled into the shipped library; this
// proves what that table does when it runs:
//
//   * GGML_SYCL_COUNTER_DUMP unset prints nothing, and set prints, per device, every counter of the
//     fixed list in order (zeros included), then one `name=<counter>{<key>}` line per key that has
//     counted, then the snapshot entries (`not_captured` until captured), and a closing
//     `[SYCL-COUNTER] end devices=N` whose N is the recorded device count;
//   * a keyed counter's total and its key lines cannot drift, and a full key table folds into
//     `(overflow)` rather than dropping a count;
//   * increments are exact under concurrency;
//   * the chokepoint classifies a refusal by the request alone: cascade_step is a counted cascade
//     miss, unconverted_ticket an interim-floor miss keyed by its ticket, anything else a terminal
//     plan refusal. Every class counts always; its line prints only under GGML_SYCL_EXT_ALLOC_TRACE=1
//     (terminal and unconverted lines once per key), and `--trace-off` proves the silent half.
//
// CPU-only by construction: nothing here allocates device memory, creates a queue or reaches the
// cache. The dump and the chokepoint are plain host code over a leaked table.

#include "ggml.h"
#include "unified-cache.hpp"

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ggml_sycl;

int g_failures = 0;

std::atomic<int> g_log_calls{ 0 };

void count_and_drop_log(ggml_log_level, const char *, void *) {
    g_log_calls.fetch_add(1);
}

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    } else {
        std::printf("ok   : %s\n", what);
    }
}

// Run `fn` with fd 2 redirected to a temporary file and return what it wrote.
template <typename Fn> std::string capture_stderr(Fn fn) {
    std::fflush(stderr);
    char tmpl[] = "/tmp/sycl-counter-dump-XXXXXX";
    int  fd     = mkstemp(tmpl);
    if (fd < 0) {
        std::fprintf(stderr, "FAIL: mkstemp\n");
        std::exit(2);
    }
    const int saved = dup(2);
    dup2(fd, 2);
    fn();
    std::fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string out;
    lseek(fd, 0, SEEK_SET);
    char    buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        out.append(buf, static_cast<size_t>(n));
    }
    close(fd);
    unlink(tmpl);
    return out;
}

std::vector<std::string> split_lines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   pos = 0;
    while (pos < s.size()) {
        size_t nl = s.find('\n', pos);
        if (nl == std::string::npos) {
            nl = s.size();
        }
        out.push_back(s.substr(pos, nl - pos));
        pos = nl + 1;
    }
    return out;
}

bool has_line(const std::vector<std::string> & lines, const std::string & want) {
    for (const auto & l : lines) {
        if (l == want) {
            return true;
        }
    }
    return false;
}

size_t index_of(const std::vector<std::string> & lines, const std::string & want) {
    for (size_t i = 0; i < lines.size(); ++i) {
        if (lines[i] == want) {
            return i;
        }
    }
    return lines.size();
}

const char * const k_counter_names[] = {
#define NAME(n) #n,
    GGML_SYCL_DUMP_COUNTERS(NAME)
#undef NAME
};

const char * const k_snapshot_names[] = {
#define NAME(id, printed) printed,
    GGML_SYCL_DUMP_SNAPSHOTS(NAME)
#undef NAME
};

}  // namespace

int main(int argc, char ** argv) {
    // Read once and cached by the library, so it is decided before anything can read it.
    const bool trace_on = !(argc > 1 && std::strcmp(argv[1], "--trace-off") == 0);
    if (trace_on) {
        setenv("GGML_SYCL_EXT_ALLOC_TRACE", "1", 1);
    } else {
        unsetenv("GGML_SYCL_EXT_ALLOC_TRACE");
    }

    // --- the raw exit's accounting --------------------------------------------------------------
    // The counters count at every raw exit, whatever the environment says: later fixtures read
    // ext_alloc_count without setting anything, and a constant 0 would make "does not move" vacuous.
    // This runs first, with GGML_SYCL_COUNTER_DUMP unset, before anything can have read and cached
    // it. Device 7 is never printed by the dump section below, so its counts do not disturb it.
    {
        static_assert(GGML_SYCL_MAX_DEVICES > 7, "the raw-exit arm uses device 7");
        unsetenv("GGML_SYCL_COUNTER_DUMP");
        const int         dev    = 7;
        const uint64_t    count0 = unified_cache_ext_alloc_count_for_testing(dev);
        const uint64_t    arena0 = unified_cache_ext_alloc_arena_count_for_testing(dev);
        const std::string out    = capture_stderr([&] {
            unified_cache_note_raw_exit(dev, 4096);
            unified_cache_note_raw_exit(dev, 8192);
            unified_cache_note_raw_exit(dev, 1);
        });
        check(unified_cache_ext_alloc_count_for_testing(dev) == count0 + 3,
              "a raw exit increments ext_alloc_count whatever the environment says");
        check(unified_cache_ext_alloc_arena_count_for_testing(dev) == arena0,
              "ext_alloc_arena does not move while the device's arena is not active");
        unified_cache_dump_arena_active_for_testing(dev, true);
        const std::string arena_out = capture_stderr([&] { unified_cache_note_raw_exit(dev, 64); });
        unified_cache_dump_arena_active_for_testing(dev, false);
        check(unified_cache_ext_alloc_count_for_testing(dev) == count0 + 4 &&
                  unified_cache_ext_alloc_arena_count_for_testing(dev) == arena0 + 1,
              "a raw exit while the arena is active also increments ext_alloc_arena");
        size_t lines_for_dev = 0;
        for (const auto & l : split_lines(out + arena_out)) {
            if (l.compare(0, 20, "[EXT-ALLOC] dev=7 ar") == 0) {
                lines_for_dev++;
            }
        }
        if (trace_on) {
            check(lines_for_dev == 4, "under the trace, each raw exit prints one [EXT-ALLOC] line (count == lines)");
        } else {
            check((out + arena_out).empty(), "without the trace a raw exit counts and prints nothing");
        }
    }

    const size_t n_counters  = sizeof(k_counter_names) / sizeof(k_counter_names[0]);
    const size_t n_snapshots = sizeof(k_snapshot_names) / sizeof(k_snapshot_names[0]);

    // --- the dump -----------------------------------------------------------------------------
    unified_cache_dump_set_device_count(2);
    unified_cache_dump_counter_add(dump_counter::ext_alloc_count, 0, 3);
    unified_cache_dump_counter_add(dump_counter::ext_alloc_count, 1, 5);
    unified_cache_dump_counter_add_key(dump_counter::set_rows_stage_arrivals, 0, "convert.cpp:48", 2);
    unified_cache_dump_counter_add_key(dump_counter::set_rows_stage_arrivals, 0, "set_rows.cpp:465");
    unified_cache_dump_counter_add_key(dump_counter::set_rows_stage_arrivals, 0, "convert.cpp:48");
    // A field with no producer on this tree is counted by nothing and prints the sentinel, so the
    // increment below (which a test can make, a shipped path cannot) must not surface as a value.
    unified_cache_dump_counter_add_key(dump_counter::load_row_op_time_arrivals, 0, "convert.cpp:48");
    unified_cache_dump_counter_add(dump_counter::onednn_graph_callback_unmarked_mallocs, 0, 2);
    unified_cache_dump_snapshot_set(dump_snapshot::zone_capacity_onednn_context_txn, 1, 4096);
    unified_cache_dump_snapshot_set_once(dump_snapshot::zone_available_weight_first_decode, 0, 111);
    unified_cache_dump_snapshot_set_once(dump_snapshot::zone_available_weight_first_decode, 0, 222);
    // Out-of-range devices are ignored, never a write past the table.
    unified_cache_dump_counter_add(dump_counter::ext_alloc_count, -1, 7);
    unified_cache_dump_counter_add(dump_counter::ext_alloc_count, 9999, 7);

    check(unified_cache_ext_alloc_count_for_testing(0) == 3, "ext_alloc_count accessor reads the counter");
    check(unified_cache_dump_counter_for_testing(dump_counter::set_rows_stage_arrivals, 0) == 4,
          "a keyed counter's total is the sum of its key counts");

    unsetenv("GGML_SYCL_COUNTER_DUMP");
    const std::string silent = capture_stderr([] { unified_cache_test_counter_dump(); });
    check(silent.empty(), "the printer prints nothing without GGML_SYCL_COUNTER_DUMP");

    setenv("GGML_SYCL_COUNTER_DUMP", "1", 1);
    const std::string dump  = capture_stderr([] { unified_cache_test_counter_dump(); });
    const auto        lines = split_lines(dump);

    check(!lines.empty() && lines.back() == "[SYCL-COUNTER] end devices=2", "the dump closes with its end line");
    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=ext_alloc_count value=3"), "dev 0 ext_alloc_count");
    check(has_line(lines, "[SYCL-COUNTER] dev=1 name=ext_alloc_count value=5"), "dev 1 ext_alloc_count");
    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=zone_plan_refusal value=0"), "a zero counter prints");
    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=set_rows_stage_arrivals value=4"),
          "keyed counter prints its total");
    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=set_rows_stage_arrivals{convert.cpp:48} value=3"),
          "key line carries its count");
    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=set_rows_stage_arrivals{set_rows.cpp:465} value=1"),
          "second key line");
    check(!has_line(lines, "[SYCL-COUNTER] dev=1 name=set_rows_stage_arrivals{convert.cpp:48} value=0"),
          "a key that has not counted on a device prints no line there");

    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=load_row_op_time_arrivals value=not_captured") &&
              has_line(lines, "[SYCL-COUNTER] dev=1 name=load_row_op_time_arrivals value=not_captured"),
          "a field whose producer has not landed prints not_captured on every device, never a zero");
    check(!has_line(lines, "[SYCL-COUNTER] dev=0 name=load_row_op_time_arrivals{convert.cpp:48} value=1"),
          "a not-captured field prints no key lines");
    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=onednn_graph_callback_unmarked_mallocs value=not_captured"),
          "the Graph callback's unmarked-malloc count (its marker lands at step 3) prints not_captured");

    // Order on dev 0: the fixed list in the table's order, then key lines, then the snapshots.
    bool in_order = true;
    for (size_t i = 0; i < n_counters; ++i) {
        const std::string want = std::string("[SYCL-COUNTER] dev=0 name=") + k_counter_names[i] + " value=";
        if (i >= lines.size() || lines[i].compare(0, want.size(), want) != 0) {
            in_order = false;
            std::fprintf(stderr, "  line %zu is `%s`, expected a `%s` line\n", i,
                         i < lines.size() ? lines[i].c_str() : "<none>", k_counter_names[i]);
        }
    }
    check(in_order, "dev 0 prints the fixed list in table order, one line per counter");
    const size_t first_key =
        index_of(lines, "[SYCL-COUNTER] dev=0 name=set_rows_stage_arrivals{convert.cpp:48} value=3");
    const size_t first_snap =
        index_of(lines, std::string("[SYCL-COUNTER] dev=0 name=") + k_snapshot_names[0] + " value=111");
    check(first_key >= n_counters && first_snap > first_key,
          "key lines follow the fixed list, snapshots follow the key lines");
    check(first_snap == index_of(lines, "[SYCL-COUNTER] dev=0 name=zone_available{WEIGHT}@first_decode value=111"),
          "a snapshot entry prints under name=<figure>@<point>");
    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=zone_largest_free{WEIGHT}@first_decode value=not_captured"),
          "a snapshot whose point was never reached prints the not_captured sentinel");
    check(has_line(lines, "[SYCL-COUNTER] dev=1 name=zone_capacity{ONEDNN}@context_txn value=4096"),
          "a captured snapshot prints its value on its device");
    check(has_line(lines, "[SYCL-COUNTER] dev=0 name=zone_capacity{ONEDNN}@context_txn value=not_captured"),
          "a snapshot is per device");
    check(index_of(lines, "[SYCL-COUNTER] dev=1 name=ext_alloc_count value=5") > first_snap,
          "device 1 follows device 0's snapshots");

    // --- the armed-only counter -----------------------------------------------------------------------
    // moe_table_reach_zero_gpu_expert evaluates only in a run whose report flag read armed, so the dump
    // marks it: not_captured until that reading is recorded armed, never a bare 0.
    {
        const char * armed_only = "[SYCL-COUNTER] dev=0 name=moe_table_reach_zero_gpu_expert value=";
        const auto   never      = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
        check(has_line(never, std::string(armed_only) + "not_captured"),
              "an armed-only counter whose flag was never read prints not_captured");
        unified_cache_dump_note_armed(false);
        check(has_line(split_lines(capture_stderr([] { unified_cache_test_counter_dump(); })),
                       std::string(armed_only) + "not_captured"),
              "an armed-only counter prints not_captured when the flag read unarmed");
        unified_cache_dump_counter_add_key(dump_counter::moe_table_reach_zero_gpu_expert, 0, "update:preload", 2);
        unified_cache_dump_note_armed(true);
        const auto armed = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
        check(has_line(armed, std::string(armed_only) + "2") &&
                  has_line(armed, "[SYCL-COUNTER] dev=0 name=moe_table_reach_zero_gpu_expert{update:preload} value=2"),
              "an armed-only counter prints its value and keys once the flag read armed");
    }

    // --- the key table ---------------------------------------------------------------------------
    for (int i = 0; i < 64; ++i) {
        char key[32];
        std::snprintf(key, sizeof(key), "site-%d", i);
        unified_cache_dump_counter_add_key(dump_counter::onednn_sdpa_fallback_after_admit, 1, key);
    }
    check(unified_cache_dump_counter_for_testing(dump_counter::onednn_sdpa_fallback_after_admit, 1) == 64,
          "a full key table still counts every call in the total");
    const auto lines2 = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
    check(has_line(lines2, "[SYCL-COUNTER] dev=1 name=onednn_sdpa_fallback_after_admit{(overflow)} value=33"),
          "keys past the table's capacity (31 named + overflow) fold into (overflow)");

    // --- concurrency -------------------------------------------------------------------------------
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([t] {
                for (int i = 0; i < 20000; ++i) {
                    unified_cache_dump_counter_add(dump_counter::onednn_sdpa_admitted, 0);
                    char key[16];
                    std::snprintf(key, sizeof(key), "k%d", (t * 7 + i) % 5);
                    unified_cache_dump_counter_add_key(dump_counter::onednn_sdpa_executed, 0, key);
                }
            });
        }
        for (auto & th : threads) {
            th.join();
        }
        check(unified_cache_dump_counter_for_testing(dump_counter::onednn_sdpa_admitted, 0) == 160000,
              "concurrent increments are exact");
        check(unified_cache_dump_counter_for_testing(dump_counter::onednn_sdpa_executed, 0) == 160000,
              "concurrent keyed increments are exact in the total");
        uint64_t   keyed = 0;
        const auto l3    = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
        for (const auto & l : l3) {
            unsigned long long v        = 0;
            static const char  prefix[] = "[SYCL-COUNTER] dev=0 name=onednn_sdpa_executed{k";
            if (l.compare(0, sizeof(prefix) - 1, prefix) == 0 &&
                std::sscanf(l.c_str() + l.rfind("value=") + 6, "%llu", &v) == 1) {
                keyed += v;
            }
        }
        check(keyed == 160000, "the key lines sum to the total under concurrency");
    }

    // --- the chokepoint ------------------------------------------------------------------------------
    // Every class counts always; its line prints only under GGML_SYCL_EXT_ALLOC_TRACE=1, which main()
    // set (or cleared, in --trace-off mode) before the first chokepoint call read it.

    // The request is constructed at each test site (`alloc_request x{}`), not here: the site label
    // is the construction line, which is what the chokepoint prints and the tests assert.
    auto fill = [](alloc_request & req, const char * cohort) {
        req.device           = 0;
        req.size             = 4096;
        req.intent.cohort_id = cohort;
    };
    auto refuse = [](const alloc_request & req, vram_zone_id zone) {
        return capture_stderr([&] { unified_cache_zone_refusal(req, zone, 4096, nullptr); });
    };
    auto count_lines = [](const std::string & text, const char * tag) {
        size_t n = 0;
        for (const auto & l : split_lines(text)) {
            if (l.compare(0, std::strlen(tag), tag) == 0) {
                n++;
            }
        }
        return n;
    };

    const uint64_t refusals0    = unified_cache_zone_plan_refusal_count_for_testing(0);
    const uint64_t cascade0     = unified_cache_zone_cascade_miss_count_for_testing(0);
    const uint64_t unconverted0 = unified_cache_zone_unconverted_miss_count_for_testing(0);

    // TERMINAL: neither flag set.
    // The construction and its expected `__LINE__` share a line, so the formatter must not split it.
    // clang-format off
    alloc_request terminal{}; const int terminal_line = __LINE__;
    // clang-format on
    fill(terminal, "chk-terminal");
    const std::string t1 = refuse(terminal, vram_zone_id::RUNTIME);
    check(unified_cache_zone_plan_refusal_count_for_testing(0) == refusals0 + 1, "terminal counts zone_plan_refusal");
    check(unified_cache_zone_cascade_miss_count_for_testing(0) == cascade0, "terminal is not a cascade miss");
    check(unified_cache_zone_unconverted_miss_count_for_testing(0) == unconverted0, "terminal is not unconverted");

    if (!trace_on) {
        check(t1.empty(), "without the trace a terminal miss counts and prints nothing");
        alloc_request cascade{};
        fill(cascade, "chk-cascade");
        cascade.intent.constraints.cascade_step = true;
        alloc_request unconverted{};
        fill(unconverted, "chk-unconverted");
        unconverted.intent.constraints.unconverted_ticket = "beni";
        const std::string quiet = refuse(cascade, vram_zone_id::RUNTIME) + refuse(unconverted, vram_zone_id::RUNTIME);
        check(quiet.empty(), "without the trace a cascade and an unconverted miss print nothing");
        check(unified_cache_zone_cascade_miss_count_for_testing(0) == cascade0 + 1 &&
                  unified_cache_zone_unconverted_miss_count_for_testing(0) == unconverted0 + 1,
              "without the trace they still count");
        const auto off = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
        check(has_line(off, "[SYCL-COUNTER] dev=0 name=zone_cascade_miss{chk-cascade} value=1") &&
                  has_line(off, "[SYCL-COUNTER] dev=0 name=zone_unconverted_miss{beni} value=1"),
              "and the dump carries their keys");
        if (g_failures != 0) {
            std::fprintf(stderr, "%d check(s) failed\n", g_failures);
            return 1;
        }
        std::printf("PASS: counter dump and zone chokepoint (trace off)\n");
        return 0;
    }

    check(count_lines(t1, "[ZONE-PLAN-BUG] dev=0 zone=RUNTIME cohort=chk-terminal site=") == 1 &&
              t1.find("test-sycl-counter-dump-printer.cpp:" + std::to_string(terminal_line)) != std::string::npos &&
              t1.find("bytes=4096") != std::string::npos && t1.find("zone_largest") == std::string::npos &&
              t1.find("zone_free") == std::string::npos,
          "terminal prints one [ZONE-PLAN-BUG] line naming the zone, cohort, construction site and bytes, and no "
          "allocator free-space figure");
    check(refuse(terminal, vram_zone_id::RUNTIME).empty() &&
              unified_cache_zone_plan_refusal_count_for_testing(0) == refusals0 + 2,
          "a repeat of the same tuple counts again and prints no second line");
    alloc_request terminal2    = terminal;
    terminal2.intent.cohort_id = "chk-terminal-2";
    check(count_lines(refuse(terminal2, vram_zone_id::RUNTIME), "[ZONE-PLAN-BUG]") == 1,
          "a different cohort is a different dedupe key");
    check(count_lines(refuse(terminal, vram_zone_id::SCRATCH), "[ZONE-PLAN-BUG]") == 1,
          "a different zone is a different dedupe key");

    // CASCADE: the next path is planned.
    alloc_request cascade{};
    fill(cascade, "chk-cascade");
    cascade.intent.constraints.cascade_step = true;
    const std::string c1                    = refuse(cascade, vram_zone_id::RUNTIME);
    check(unified_cache_zone_cascade_miss_count_for_testing(0) == cascade0 + 1, "cascade counts zone_cascade_miss");
    check(unified_cache_zone_plan_refusal_count_for_testing(0) == refusals0 + 4,
          "a cascade miss is not a plan refusal");
    check(count_lines(c1, "[ZONE-CASCADE] dev=0 zone=RUNTIME cohort=chk-cascade") == 1 &&
              c1.find("ZONE-PLAN-BUG") == std::string::npos,
          "a cascade miss prints a [ZONE-CASCADE] line and no plan-bug line");
    const auto l4 = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
    check(has_line(l4, "[SYCL-COUNTER] dev=0 name=zone_cascade_miss{chk-cascade} value=1"),
          "a cascade miss is keyed by its cohort");

    // A cohort-less request is named by its construction site.
    // clang-format off
    alloc_request nameless{}; const int nameless_line = __LINE__;
    // clang-format on
    fill(nameless, nullptr);
    nameless.intent.constraints.cascade_step = true;
    (void) refuse(nameless, vram_zone_id::RUNTIME);
    const auto l5 = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
    check(has_line(l5, "[SYCL-COUNTER] dev=0 name=zone_cascade_miss{test-sycl-counter-dump-printer.cpp:" +
                           std::to_string(nameless_line) + "} value=1"),
          "a cohort-less request uses its site as the cohort");

    // UNCONVERTED: an interim-floor row whose exact term another ticket owns.
    alloc_request unconverted{};
    fill(unconverted, "chk-unconverted");
    unconverted.intent.constraints.unconverted_ticket = "beni";
    const std::string u1                              = refuse(unconverted, vram_zone_id::RUNTIME);
    const std::string u2                              = refuse(unconverted, vram_zone_id::RUNTIME);
    check(unified_cache_zone_unconverted_miss_count_for_testing(0) == unconverted0 + 2,
          "unconverted counts every miss");
    check(unified_cache_zone_plan_refusal_count_for_testing(0) == refusals0 + 4,
          "an unconverted miss is not a plan refusal");
    check(count_lines(u1, "[ZONE-UNCONVERTED] dev=0 zone=RUNTIME cohort=chk-unconverted") == 1 &&
              u1.find("ticket=beni") != std::string::npos && u2.empty(),
          "unconverted prints one line per key naming its ticket");
    const auto l6 = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
    check(has_line(l6, "[SYCL-COUNTER] dev=0 name=zone_unconverted_miss{beni} value=2"),
          "an unconverted miss is keyed by its ticket");

    // cascade_step wins over a ticket: the request says its next path is planned.
    alloc_request both{};
    fill(both, "chk-both");
    both.intent.constraints.cascade_step       = true;
    both.intent.constraints.unconverted_ticket = "pqmm";
    const std::string b1                       = refuse(both, vram_zone_id::RUNTIME);
    check(unified_cache_zone_unconverted_miss_count_for_testing(0) == unconverted0 + 2 &&
              b1.find("ZONE-UNCONVERTED") == std::string::npos,
          "a request that is both a cascade step and unconverted counts as the cascade step");

    // The 128-entry dedupe table says so when full, and never drops the count.
    const uint64_t before_full = unified_cache_zone_plan_refusal_count_for_testing(0);
    std::string    flood_out;
    for (int i = 0; i < 200; ++i) {
        char cohort[32];
        std::snprintf(cohort, sizeof(cohort), "chk-flood-%d", i);
        alloc_request flood{};
        fill(flood, cohort);
        flood_out += refuse(flood, vram_zone_id::RUNTIME);
    }
    check(unified_cache_zone_plan_refusal_count_for_testing(0) == before_full + 200,
          "a flood of distinct tuples counts every refusal");
    check(flood_out.find("(dedupe-full)") != std::string::npos,
          "a full dedupe table prints the line with (dedupe-full), never silently");

    // --- report lines and snapshot control -----------------------------------------------------------
    {
        unsetenv("GGML_SYCL_COUNTER_DUMP");
        check(!unified_cache_dump_report_enabled(), "reports are off without GGML_SYCL_COUNTER_DUMP");
        check(capture_stderr([] { unified_cache_dump_report("landing dev=0"); }).empty() &&
                  capture_stderr([] { (void) unified_cache_dump_report_once("k-off", "landing dev=0"); }).empty(),
              "a report prints nothing while the dump is not armed");
        setenv("GGML_SYCL_COUNTER_DUMP", "1", 1);
        check(unified_cache_dump_report_enabled(), "reports are on under GGML_SYCL_COUNTER_DUMP=1");
        const auto r1 = split_lines(capture_stderr([] { unified_cache_dump_report("landing dev=0 size=1"); }));
        check(r1.size() == 1 && r1[0] == "[SYCL-REPORT] landing dev=0 size=1",
              "a report prints as [SYCL-REPORT] <text>");

        // Default verbosity drops GGML_LOG_INFO in every tool (common_log maps it above the threshold), so
        // a report that went through the log would vanish from a plain llama-completion run and leave G0
        // vacuous. Install a log sink that swallows everything: the report and the dump must still reach
        // fd 2, and must not have touched the log at all.
        {
            g_log_calls = 0;
            ggml_log_set(count_and_drop_log, nullptr);
            // Positive control: the sink does see a log line, so a zero below means something.
            GGML_LOG_WARN("[SYCL-REPORT-TEST] sink control\n");
            const bool sink_live = g_log_calls == 1;
            g_log_calls          = 0;
            const auto bypass_report =
                split_lines(capture_stderr([] { unified_cache_dump_report("landing dev=0 log=bypass"); }));
            const auto bypass_dump = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
            ggml_log_set(nullptr, nullptr);
            check(bypass_report.size() == 1 && bypass_report[0] == "[SYCL-REPORT] landing dev=0 log=bypass",
                  "a report reaches stderr with every log level swallowed");
            check(!bypass_dump.empty() && bypass_dump.back().rfind("[SYCL-COUNTER] end devices=", 0) == 0,
                  "the counter dump reaches stderr with every log level swallowed");
            check(sink_live, "the log sink counts a line that does go through the log");
            check(g_log_calls == 0, "neither the report nor the dump goes through the ggml log");
        }
        const auto o1 = split_lines(
            capture_stderr([] { (void) unified_cache_dump_report_once("arm_a:0:Q6_K:512", "arm_a_kernel ne11=512"); }));
        const auto o2 = split_lines(
            capture_stderr([] { (void) unified_cache_dump_report_once("arm_a:0:Q6_K:512", "arm_a_kernel ne11=512"); }));
        const auto o3 = split_lines(
            capture_stderr([] { (void) unified_cache_dump_report_once("arm_a:0:Q6_K:16", "arm_a_kernel ne11=16"); }));
        check(o1.size() == 1 && o2.empty() && o3.size() == 1,
              "report_once prints a key's first reading only, and a distinct key prints again");
        bool       full_reported = true;
        const auto flood         = split_lines(capture_stderr([&] {
            for (int i = 0; i < 80; ++i) {
                char key[32];
                std::snprintf(key, sizeof(key), "flood-%d", i);
                full_reported = unified_cache_dump_report_once(key, "flood") && full_reported;
            }
        }));
        size_t     announced     = 0;
        for (const auto & l : flood) {
            announced += l.rfind("[SYCL-REPORT] (report-once-full)", 0) == 0 ? 1 : 0;
        }
        check(!full_reported, "a full report_once table says so (returns false), never drops silently");
        check(announced == 1, "a full report_once table announces itself once, on the report path");

        // set / pending / clear: the load-end entries overwrite, and load_2 returns to not_captured.
        check(unified_cache_dump_snapshot_pending(dump_snapshot::weight_planned_device_bytes_load_2, 0),
              "an untouched snapshot is pending");
        unified_cache_dump_snapshot_set(dump_snapshot::weight_planned_device_bytes_load_2, 0, 77);
        unified_cache_dump_snapshot_set(dump_snapshot::weight_planned_device_bytes_load_2, 0, 88);
        const auto s1 = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
        check(has_line(s1, "[SYCL-COUNTER] dev=0 name=weight_planned_device_bytes{load_2}@last_load_end value=88"),
              "set overwrites the previous capture");
        unified_cache_dump_snapshot_clear(dump_snapshot::weight_planned_device_bytes_load_2, 0);
        const auto s2 = split_lines(capture_stderr([] { unified_cache_test_counter_dump(); }));
        check(has_line(s2,
                       "[SYCL-COUNTER] dev=0 name=weight_planned_device_bytes{load_2}@last_load_end "
                       "value=not_captured") &&
                  unified_cache_dump_snapshot_pending(dump_snapshot::weight_planned_device_bytes_load_2, 0),
              "clear returns a snapshot to not_captured and pending");
        // set_once after a clear captures again (the first_decode claim is released with the value).
        unified_cache_dump_snapshot_set_once(dump_snapshot::weight_planned_device_bytes_load_2, 0, 5);
        check(!unified_cache_dump_snapshot_pending(dump_snapshot::weight_planned_device_bytes_load_2, 0),
              "a claimed once-only snapshot is no longer pending");
    }

    (void) n_snapshots;

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("PASS: counter dump and zone chokepoint\n");
    return 0;
}
