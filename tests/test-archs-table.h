#pragma once

// The results table of test-llama-archs: how a row is assembled, how it reaches stdout, and
// what a well-formed line looks like. The three live together because they are one contract,
// and because tests/test-archs-table.cpp gates the emitter THROUGH this header rather
// than re-stating the layout -- a parser carrying its own copy of the format stops testing the
// harness the moment either copy drifts.
//
// Why emission needs a rule at all (llama.cpp-to9m). A sweep is read with stdout and stderr
// merged, because that is where the backend's log lines are. The harness used to print a row in
// two halves -- prefix, fflush(stdout), load a model and decode, then the result -- so every
// line logged in between landed INSIDE the row. Measured over four sweeps on 2026-08-01
// (artifacts/1p2k/): 6 to 8 of the 50 B70 rows mangled, and EVERY FAIL cell in EVERY run was
// among the mangled ones. (The ticket first said 7 to 11 of 51. That denominator came from
// `grep -c B70`, which also counts the stderr detail lines the table deliberately writes out of
// band -- one of them in s7-a -- so every subtraction was one high. `--check` counts in band and
// cannot drift that way.) That is not chance: a failing row took the slow or odd path, and
// that path is what logs. `grep FAIL` on such output comes back clean on a run that failed --
// a check that fails open. A row is now assembled whole and written with one stdio call.

#include "common.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>

struct archs_table {
    // A well-formed line carries exactly this many pipe-delimited fields, and therefore
    // n_columns + 1 '|' characters: arch, device, config, NMSE, roundtrip.
    static constexpr size_t n_columns = 5;

    size_t width_arch   = 4;
    size_t width_device = 4;

    // Header line, separator, and the legend.
    //
    // The legend is printed unconditionally, and it has to be. The thing it exists to prevent is
    // someone reading an all-OK Roundtrip column as proof that the logits matched bit for bit.
    // That is precisely the run in which a mismatch-triggered notice would not appear, so a
    // conditional notice is absent exactly when it is needed.
    //
    // Its lines start at column 0, not with '|': they are not rows, and a strict row parser must
    // skip them rather than report them as malformed.
    std::string header(const double nmse_gate) const {
        std::string text = string_format(
            ("|%" + std::to_string(width_arch) + "s|%" + std::to_string(width_device) + "s|%6s|%15s|%9s|\n").c_str(),
            "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
        text += "|" + std::string(width_arch, '-') + "|" + std::string(width_device, '-') +
                "|------|---------------|---------|\n";
        text += "Roundtrip: a GGUF save+reload of the device model, compared against that model.\n";
        text += string_format("  Gated on NMSE(device, reloaded) <= %.0e, NOT on bit-equality: this path is not\n",
                              nmse_gate);
        text += "  bit-reproducible run to run, so OK does NOT mean the logits matched bit for bit.\n";
        text += "  BITDIFF = bits differ, within tolerance. Magnitudes for any non-bit-exact row\n";
        text += "  are on stderr.\n";
        return text;
    }

    // One row. The status strings carry ANSI colour, so the field widths below pad the escape
    // sequences too and the columns do not line up with the header's; that is pre-existing and
    // deliberately left alone -- the parser keys on the delimiters, not on alignment.
    std::string row(const char * arch,
                    const char * device,
                    const char * config,
                    const char * status_nmse,
                    const char * nmse_value,
                    const char * status_roundtrip) const {
        return string_format(
            ("|%" + std::to_string(width_arch) + "s|%" + std::to_string(width_device) + "s|%6s|%15s %10s|%20s|\n")
                .c_str(),
            arch, device, config, status_nmse, nmse_value, status_roundtrip);
    }

    // Unconditional, like the legend, and for the same reason -- a reader who scrolls to the
    // bottom of a green run must still be told what OK did and did not establish. Stating the
    // count even when it is zero is the point: "0 rows" is a measurement, while silence is
    // indistinguishable from the check not having run.
    std::string footer(const double nmse_gate, const size_t n_bitdiff) const {
        return string_format(
            "Roundtrip gate: NMSE(device, reloaded) <= %.0e, not bit-equality. "
            "%zu row(s) round-tripped within tolerance but not bit-for-bit.\n",
            nmse_gate, n_bitdiff);
    }

    // One table line, one stdio call.
    //
    // stdout is fully buffered when it is a file or a pipe, so a partial flush can split a row at
    // any byte; stderr is unbuffered, so whatever the backend logs in the gap lands between the
    // halves. Flushing first leaves the buffer empty, so a line shorter than the buffer cannot
    // straddle a boundary; the single fputs puts all of it in; flushing after makes it one
    // write(2), which the kernel does not interleave with the log's writes to the shared file
    // description. Callers must never build a table line from successive printf()s.
    static void emit(const std::string & text) {
        fflush(stdout);
        fputs(text.c_str(), stdout);
        fflush(stdout);
    }

    // A line the table is responsible for. Anything else on the stream -- log output, the
    // legend, the footer, the stderr breadcrumbs -- is not this parser's business.
    static bool line_is_table(const std::string & line) { return !line.empty() && line[0] == '|'; }

    static bool line_is_well_formed(const std::string & line) {
        if (!line_is_table(line) || line.back() != '|') {
            return false;
        }
        return (size_t) std::count(line.begin(), line.end(), '|') == n_columns + 1;
    }

    // The other half of a split row: the result fields, stranded on a line of their own because
    // log output ended the line the prefix was on. Detected separately from a malformed row
    // because a split produces BOTH, and a check that counted only malformed '|' lines would miss
    // any split whose prefix happened to still parse.
    static bool line_is_orphan_tail(const std::string & line) {
        return !line.empty() && line[0] != '|' && line.back() == '|';
    }
};
