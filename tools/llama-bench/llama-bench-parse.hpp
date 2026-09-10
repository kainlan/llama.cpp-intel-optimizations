#pragma once

// llama.cpp-y8xv quality round 1, Q5: parse_int_range() and
// parse_ubatch_range() hoisted out of llama-bench.cpp into a header-only
// unit, byte-identical to what llama-bench.cpp defined inline, so that
// tests/test-llama-bench-parse.cpp can exercise them without linking or
// building llama-bench itself (which drags in the whole llama/ggml stack).
// llama-bench.cpp includes this header instead of defining these functions
// itself. Neither function depends on anything outside the standard
// library.

#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<int> parse_int_range(const std::string & s, bool allow_negative = false) {
    // first[-last[(+|*)step]]
    std::regex range_regex(allow_negative
        ? R"(^(-?\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))"
        : R"(^(\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))");

    std::smatch match;
    std::string::const_iterator search_start(s.cbegin());
    std::vector<int> result;
    while (std::regex_search(search_start, s.cend(), match, range_regex)) {
        int  first = std::stoi(match[1]);
        int  last  = match[2].matched ? std::stoi(match[2]) : first;
        char op    = match[3].matched ? match[3].str()[0] : '+';
        int  step  = match[4].matched ? std::stoi(match[4]) : 1;

        for (int i = first; i <= last;) {
            result.push_back(i);

            int prev_i = i;

            if (op == '+') {
                i += step;
            } else if (op == '*') {
                i *= step;
            } else {
                throw std::invalid_argument("invalid range format");
            }

            if (i <= prev_i) {
                throw std::invalid_argument("invalid range");
            }
        }
        search_start = match.suffix().first;
    }

    if (search_start != s.cend()) {
        throw std::invalid_argument("invalid range format");
    }

    return result;
}

// llama.cpp-nphx: -ub/--ubatch-size accepts the literal token "auto" (any
// number of times, comma-separated with ordinary integers) as a stand-in for
// the sentinel -1, which cmd_params_instance::to_llama_cparams() below turns
// into n_ubatch_auto=true. Only the exact token "auto" maps to that sentinel;
// every other comma-separated token is parsed by the UNMODIFIED, non-negative
// parse_int_range() (the same call the base -ub handler used), so a literal
// negative number (e.g. "-256", or "-1" typed directly rather than via
// "auto") is rejected with the identical "invalid range format" base already
// throws -- allow_negative is never turned on here, which is what keeps that
// rejection intact; passing allow_negative=true through to a comma-joined
// string would have let ANY negative token silently through as if it were
// "auto" (llama.cpp-y8xv spec round 1, F3).
static std::vector<int> parse_ubatch_range(const std::string & s) {
    std::vector<int>  result;
    std::string       tok;
    std::stringstream ss(s);
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
            // llama.cpp-y8xv spec round 2, F7: a leading or doubled comma
            // (e.g. ",256" or "256,,512") yields an empty token here; base's
            // single parse_int_range() call over the whole string rejected
            // both the same way it rejects any other malformed input, so
            // reject it here too rather than silently dropping it. A
            // trailing comma ("512,") never reaches this branch: getline
            // stops returning tokens once the stream is exhausted, so it
            // still produces exactly one token, matching base's accept.
            throw std::invalid_argument("invalid range format");
        }
        if (tok == "auto") {
            result.push_back(-1);
            continue;
        }
        auto p = parse_int_range(tok);
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
}

// llama.cpp-y8xv quality round 2, R1: whether the markdown/CSV/JSON printer
// should show the n_ubatch column. The base condition (differs from the
// tool's own default, or more than one value was requested) is not enough
// on its own: under GGML_USE_SYCL cmd_params_defaults.n_ubatch is {-1} (the
// auto sentinel), so a bare run or an explicit "-ub auto" has
// requested_n_ubatch == defaults_n_ubatch and the base condition alone would
// never show the column -- exactly the case whose RESOLVED value (see the
// GGML_SYCL_AUTO_UBATCH doc row) is the whole point of printing it. Off
// SYCL, defaults_n_ubatch never contains -1, so this predicate reduces to
// the base condition and nothing changes there.
static bool bench_prints_n_ubatch_column(const std::vector<int> & requested_n_ubatch,
                                         const std::vector<int> & defaults_n_ubatch) {
    if (requested_n_ubatch.size() > 1 || requested_n_ubatch != defaults_n_ubatch) {
        return true;
    }
    return std::find(requested_n_ubatch.begin(), requested_n_ubatch.end(), -1) != requested_n_ubatch.end();
}
