// llama.cpp-y8xv quality round 1, Q5: host-only unit test for
// parse_int_range() / parse_ubatch_range(), hoisted out of
// tools/llama-bench/llama-bench.cpp into llama-bench-parse.hpp precisely so
// this test can exercise them without linking or building llama-bench
// itself. This test is the evidence of record for the -ub range-parser
// behaviour that spec rounds 1 (F3) and 2 (F7) previously proved with a
// standalone Python port of the same logic -- see those commits' bodies
// (e71df28dd, b44d3262d) for the derivation; this test pins the same eight
// rows plus one more ("-1" rejected) directly against the real C++ code.

#include "llama-bench-parse.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static bool throws_invalid_argument(const std::string & s) {
    try {
        parse_ubatch_range(s);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

static void test(void) {
    // Rows from b44d3262d's commit body table (base vs round 1 vs round 2),
    // plus a bare "-1" (typed directly rather than via "auto"), which must
    // be rejected the same way "-256" is (F3: only the literal token "auto"
    // may produce the -1 sentinel).
    assert(throws_invalid_argument(","));
    assert(throws_invalid_argument(",256"));
    assert(throws_invalid_argument("256,,512"));
    assert(throws_invalid_argument("-256"));
    assert(throws_invalid_argument("-1"));

    {
        // Trailing comma: still accepted, matching base's parse_int_range()
        // called directly on the whole string (F7 must not regress this).
        auto p = parse_ubatch_range("512,");
        assert(p.size() == 1);
        assert(p[0] == 512);
    }
    {
        // "auto" alone maps to the -1 sentinel.
        auto p = parse_ubatch_range("auto");
        assert(p.size() == 1);
        assert(p[0] == -1);
    }
    {
        // "auto" mixed with an ordinary value, comma-separated.
        auto p = parse_ubatch_range("auto,256");
        assert(p.size() == 2);
        assert(p[0] == -1);
        assert(p[1] == 256);
    }
    {
        // An ordinary range expression still expands via the unmodified
        // parse_int_range() underneath.
        auto p = parse_ubatch_range("512-1024");
        assert(p.size() == 513);
        assert(p.front() == 512);
        assert(p.back() == 1024);
        for (size_t i = 0; i < p.size(); i++) {
            assert(p[i] == 512 + static_cast<int>(i));
        }
    }

    // parse_int_range() itself (used directly elsewhere in llama-bench.cpp,
    // e.g. -ngl with allow_negative=true) keeps its own two behaviours this
    // header must not disturb: non-negative by default, and a step form.
    {
        auto p = parse_int_range("4");
        assert(p.size() == 1 && p[0] == 4);
    }
    {
        bool threw = false;
        try {
            parse_int_range("-4");
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
    }
    {
        auto p = parse_int_range("-4", /*allow_negative=*/true);
        assert(p.size() == 1 && p[0] == -4);
    }
    {
        auto             p      = parse_int_range("1-8*2");
        std::vector<int> expect = { 1, 2, 4, 8 };
        assert(p == expect);
    }

    // llama.cpp-y8xv quality round 2, R1: bench_prints_n_ubatch_column().
    {
        // The auto sentinel alone must always show the column, even when it
        // is (trivially) equal to a one-element default of {-1} -- this is
        // the SYCL bare-run / "-ub auto" case the base condition missed.
        assert(bench_prints_n_ubatch_column({ -1 }, { -1 }) == true);
    }
    {
        // Matches the tool's own (non-SYCL) default exactly, no -1 present:
        // no reason to show the column.
        assert(bench_prints_n_ubatch_column({ 512 }, { 512 }) == false);
    }
    {
        // More than one requested value: always show it, regardless of
        // whether it happens to equal the default.
        assert(bench_prints_n_ubatch_column({ 512, 1024 }, { 512 }) == true);
    }
    {
        // Differs from the default outright (base condition, unaffected).
        assert(bench_prints_n_ubatch_column({ 1024 }, { 512 }) == true);
    }

    printf("test-llama-bench-parse: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-llama-bench-parse: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
