// Gate for -x/--exclude of test-llama-archs (llama.cpp-fepd).
//
// The sweep this serves cannot be run to check it: test-llama-archs loads models onto a GPU, it
// is on this project's never-loop list, and a single run peaks near 200 GB of 255 GB. So the
// gate drives tests/test-archs-exclude.h directly -- the same definitions the harness calls,
// not a second copy of them -- and loads no model, initialises no backend and touches no
// device. It links llama only for llm_arch_from_string()/llm_arch_all(), which are pure table
// lookups.
//
// What it pins, in the language of the ticket's acceptance criteria:
//   (2) -x is repeatable and accumulates                -> `parse accumulates`
//   (3) EXCLUDED is distinguishable from SKIP and OK,
//       and is a ROW rather than an absence              -> `status word is disjoint`, `row ...`
//   (4) an unknown arch name is an error                 -> `parse rejects ...`
//   (5) -x absent leaves behaviour unchanged             -> `empty set excludes nothing`
//
// Positive controls, because a green check whose RED was never observed is worth nothing: every
// predicate below is also handed an input for which it must FAIL, and the gate fails if it does
// not. That is not ceremony here -- three probes on this ticket's sibling tasks returned clean,
// plausible zeros while being structurally incapable of firing.

#include "test-archs-exclude.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_n_fail = 0;

// Not assert(): -DNDEBUG erases it, and this gate has to fail a release build too.
static void check(const bool ok, const char * what) {
    if (ok) {
        fprintf(stderr, "ok   : %s\n", what);
        return;
    }
    fprintf(stderr, "FAIL : %s\n", what);
    g_n_fail++;
}

static bool has_substr(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

// A table line as archs_table's parser sees it: emit() writes rows with a trailing newline, the
// parser works on lines without one.
static std::string chomp(const std::string & line) {
    return !line.empty() && line.back() == '\n' ? line.substr(0, line.size() - 1) : line;
}

// The other statuses this table already uses. Only their WORDS are relied on, never their
// colours, so this list cannot go stale in a way that makes the check below pass vacuously: it
// is a claim about what a reader (or a grep) must be able to tell apart, and recolouring SKIP
// upstream would not change any of these four strings.
static const char * g_other_status_words[] = { "OK", "FAIL", "SKIP", "BITDIFF" };

int main() {
    // ---- criterion 4: an unknown name is an error -------------------------------------------
    {
        std::vector<llm_arch> archs;
        check(archs_exclude::parse("llama", archs), "parse accepts a real architecture name");
        check(archs.size() == 1 && archs[0] == LLM_ARCH_LLAMA, "parse appends the resolved architecture");

        // Positive control for the two rejections below: the accept above is what proves parse()
        // can return true at all, so a false from it is a decision and not a stuck predicate.
        const size_t n_before = archs.size();
        check(!archs_exclude::parse("__no_such_arch__", archs), "parse rejects an unknown name");
        check(archs.size() == n_before, "a rejected name does not grow the exclude set");

        // "(unknown)" is a real entry in LLM_ARCH_NAMES, so llm_arch_from_string() resolves it
        // rather than failing to find it -- and it resolves to the one arch the sweep skips
        // before the exclusion check runs. Excluding it would therefore be a silent no-op that
        // still looked accepted.
        check(llm_arch_from_string("(unknown)") == LLM_ARCH_UNKNOWN,
              "control: '(unknown)' really is a resolvable name, so the case below is not hypothetical");
        check(!archs_exclude::parse("(unknown)", archs), "parse rejects the '(unknown)' spelling too");
        check(archs.size() == n_before, "'(unknown)' does not grow the exclude set either");
    }

    // ---- criterion 2: -x is repeatable and accumulates ---------------------------------------
    {
        std::vector<llm_arch> archs;
        check(archs_exclude::parse("gemma2", archs) && archs_exclude::parse("deepseek2", archs),
              "parse accepts a second -x");
        check(archs.size() == 2, "parse accumulates rather than replacing");
        check(archs_exclude::contains(archs, LLM_ARCH_GEMMA2) && archs_exclude::contains(archs, LLM_ARCH_DEEPSEEK2),
              "both accumulated architectures are excluded");
        // The payoff case of the ticket, stated as data: gemma2 is where every sweep dies
        // (llama.cpp-4lnb) and deepseek2 is declared after it, so a sweep that excludes the
        // first is the only way anyone has ever been able to reach the second.
        check(LLM_ARCH_DEEPSEEK2 > LLM_ARCH_GEMMA2,
              "control: deepseek2 really is declared after gemma2, which is why -x gemma2 is the unblocker");

        check(archs_exclude::parse("gemma2", archs) && archs.size() == 3,
              "a repeated -x is accepted rather than rejected");
        check(archs_exclude::contains(archs, LLM_ARCH_GEMMA2), "a repeated -x still excludes");
    }

    // ---- criterion 5: no -x means nothing is excluded ----------------------------------------
    {
        const std::vector<llm_arch> none;
        size_t                      n_contained = 0;
        for (const llm_arch & arch : llm_arch_all()) {
            n_contained += archs_exclude::contains(none, arch);
        }
        check(n_contained == 0, "an empty exclude set excludes no architecture at all");
        // Positive control: the same loop over a one-element set must find exactly one, so the
        // zero above is a measurement and not a loop that never ran.
        const std::vector<llm_arch> one = { LLM_ARCH_LLAMA };
        n_contained                     = 0;
        for (const llm_arch & arch : llm_arch_all()) {
            n_contained += archs_exclude::contains(one, arch);
        }
        check(n_contained == 1, "control: the same sweep over a one-element set finds exactly one");
    }

    // ---- criterion 3: EXCLUDED is its own status, and it is a ROW ----------------------------
    {
        const std::string status = archs_exclude::status;
        check(has_substr(status, "EXCLUDED"), "the excluded status says EXCLUDED");
        for (const char * other : g_other_status_words) {
            check(!has_substr(status, other) && !has_substr(std::string(other), "EXCLUDED"),
                  (std::string("EXCLUDED and ") + other + " are disjoint words").c_str());
        }
        // Positive control: the disjointness predicate must be able to report an overlap.
        check(has_substr(status, "CLUDE"), "control: the substring test does fire on an overlap");
    }

    // ---- criterion 3: the row is well formed and survives the atomic-write bound -------------
    {
        // Worst case as the harness will actually build it: the longest architecture name that
        // exists, and the longest device description this fork has seen ("Intel(R) Arc(TM) Pro
        // B70 Graphics"). Derived from llm_arch_all() rather than written down, so a longer
        // arch name added upstream is measured here instead of silently overflowing.
        // Asserted BEFORE the loop that consumes it, not after. An empty arch list would leave
        // width_arch at 0 and longest at "", and every check below would then be measuring a
        // row built from nothing -- and would pass, vacuously. A control placed after the code
        // it protects cannot prevent that; it can only explain it afterwards.
        check(!llm_arch_all().empty(), "control: llm_arch_all() is non-empty before anything reads it");

        size_t       width_arch = 0;
        const char * longest    = "";
        for (const llm_arch & arch : llm_arch_all()) {
            if (strlen(llm_arch_name(arch)) > width_arch) {
                width_arch = strlen(llm_arch_name(arch));
                longest    = llm_arch_name(arch);
            }
        }
        // A separate claim from the one above, not a repeat of it: the list is non-empty AND at
        // least one name is a non-empty string, so the width the row is built at is real.
        check(width_arch > 0, "the longest architecture name has non-zero length");

        const archs_table table = { width_arch, 33 };
        const std::string line  = archs_exclude::row(table, longest);

        check(archs_table::line_is_well_formed(chomp(line)), "the EXCLUDED row parses as a well-formed table row");
        check(!archs_table::line_is_orphan_tail(chomp(line)), "the EXCLUDED row is not an orphan tail");
        check(line.size() < archs_table::max_atomic_line,
              "the EXCLUDED row fits the atomic-write bound emit() promises");

        // Positive controls for both probes above.
        const archs_table absurd = { archs_table::max_atomic_line, archs_table::max_atomic_line };
        check(archs_exclude::row(absurd, longest).size() >= archs_table::max_atomic_line,
              "control: the length probe does fire on an over-long row");
        check(!archs_table::line_is_well_formed(chomp(line) + "|"),
              "control: the well-formedness probe does fire on a mangled row");

        // The row is what makes an exclusion visible. An absent row is indistinguishable from
        // one that was never reached -- which is precisely what a sweep dying at gemma2 already
        // produces -- so the row must name the architecture as well as its status.
        check(has_substr(line, longest) && has_substr(line, "EXCLUDED"),
              "the row names both the architecture and its EXCLUDED status");
    }

    fprintf(stderr, "%s: %d check(s) failed\n", __func__, g_n_fail);
    return g_n_fail == 0 ? 0 : 1;
}
