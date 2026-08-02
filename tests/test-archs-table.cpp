// Gate for the results table of test-llama-archs (llama.cpp-to9m).
//
// The property under test: on a stream where the table and the backend's log output are merged
// -- `test-llama-archs > sweep.log 2>&1`, which is how every captured sweep is read -- every
// line the table emits still parses as a table line. Before this gate existed, log lines landed
// inside half-written rows and the mangled rows were, in all four captured sweeps, exactly the
// rows carrying FAIL. `grep FAIL` returned clean on runs that had failed.
//
// Everything here is CPU-only and device-free: it drives the emitter from tests/
// test-archs-table.h directly, so no model is loaded and no backend is initialised.
//
// Positive controls, because a green gate whose RED was never observed is worth nothing:
//   * `pre-fix emission` replays the exact sequence this ticket removed (prefix, fflush, log,
//     result) through the same capture and the same parser, and the gate FAILS if the parser
//     calls that output clean.
//   * `historical sweep excerpt` is copied verbatim out of artifacts/1p2k/1p2k-s7-a.txt -- real
//     pre-fix output from a real run -- and must likewise be seen as mangled.
//   * the fixed-emission case asserts the log noise is PRESENT in its capture, so a run that
//     silently applied no interleaving pressure cannot pass by having nothing to corrupt.
//
// Run with `--check <file>` to point the same parser at a captured sweep log.

#include "test-archs-table.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <io.h>
#    define archs_dup    _dup
#    define archs_dup2   _dup2
#    define archs_close  _close
#    define archs_fileno _fileno
#else
#    include <unistd.h>
#    define archs_dup    dup
#    define archs_dup2   dup2
#    define archs_close  close
#    define archs_fileno fileno
#endif

// Widths as a real sweep produces them (artifacts/1p2k/): the longest arch name, and
// "Intel(R) Arc(TM) Pro B70 Graphics".
static const archs_table g_table = { 16, 33 };

static const double g_nmse_gate = 1e-4;

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

struct parse_result {
    size_t n_table       = 0;  // lines starting with '|'
    size_t n_well_formed = 0;
    size_t n_mangled     = 0;
    size_t n_orphan_tail = 0;
};

static std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::string              line;
    for (const char c : text) {
        if (c == '\n') {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            lines.push_back(line);
            line.clear();
            continue;
        }
        line += c;
    }
    if (!line.empty()) {
        lines.push_back(line);
    }
    return lines;
}

static parse_result parse_table(const std::string & text, const bool report) {
    parse_result                   res;
    const std::vector<std::string> lines = split_lines(text);
    for (size_t i = 0; i < lines.size(); i++) {
        const std::string & line = lines[i];
        if (archs_table::line_is_table(line)) {
            res.n_table++;
            if (archs_table::line_is_well_formed(line)) {
                res.n_well_formed++;
            } else {
                res.n_mangled++;
                if (report) {
                    fprintf(stderr, "  line %zu: mangled row: %s\n", i + 1, line.c_str());
                }
            }
        } else if (archs_table::line_is_orphan_tail(line)) {
            res.n_orphan_tail++;
            if (report) {
                fprintf(stderr, "  line %zu: orphaned row tail: %s\n", i + 1, line.c_str());
            }
        }
    }
    return res;
}

// Rows that a strict parser accepts AND that carry `needle`. This is the quantity the ticket is
// actually about: `grep FAIL` over the table is only a check if the FAIL rows survive parsing.
static size_t n_well_formed_with(const std::string & text, const char * needle) {
    size_t                         n     = 0;
    const std::vector<std::string> lines = split_lines(text);
    for (size_t i = 0; i < lines.size(); i++) {
        if (archs_table::line_is_well_formed(lines[i]) && lines[i].find(needle) != std::string::npos) {
            n++;
        }
    }
    return n;
}

// What a model load and decode put on stderr while a row is being produced. Two real lines,
// copied from artifacts/1p2k/1p2k-s7-a.txt, one of them the one that actually split a row there.
static void log_noise() {
    fprintf(stderr, "[SYCL] GGML_SYCL_F16 build: attention Q/accumulators are f16\n");
    fprintf(stderr,
            "0.02.114.321 E [UNIFIED-CACHE] planned-mode runtime materialization rejected"
            " op=direct_stage_weight caller=direct_stage_weight graph_active=1\n");
}

struct row_case {
    const char * arch;
    const char * device;
    const char * config;
    const char * status_nmse;
    const char * nmse_value;
    const char * status_roundtrip;
};

// Every status the harness can print, including the ones a mangled table hides: FAIL, the
// non-finite "(nan)" value, and BITDIFF.
// clang-format off
static const row_case g_rows[] = {
    { "llama",  "Intel(R) Arc(TM) Pro B70 Graphics", "Dense", "\033[1;32mOK\033[0m",   "(1.07e-10)", "\033[1;32mOK\033[0m"      },
    { "clip",   "Intel(R) Core(TM) Ultra 7 265K",    "Dense", "\033[1;33mSKIP\033[0m", "",           "\033[1;33mSKIP\033[0m"    },
    { "gemma2", "Intel(R) Arc(TM) Pro B70 Graphics", "MoE",   "\033[1;31mFAIL\033[0m", "(nan)",      "\033[1;31mFAIL\033[0m"    },
    { "qwen3",  "Meta",                              "MoE",   "\033[1;32mOK\033[0m",   "(2.20e-09)", "\033[1;36mBITDIFF\033[0m" },
};
// clang-format on

static const size_t g_n_rows = sizeof(g_rows) / sizeof(g_rows[0]);

// The widest table a real sweep could plausibly produce: the longest arch name in llm_arch_all()
// against a device description far longer than any this fork has seen. If a row at THAT size
// still fits emit()'s atomic-write bound, the bound is not being met by luck.
static const archs_table g_table_wide = { 40, 80 };

// Wide enough that a row cannot fit the bound. Used only as the control that proves the bound is
// reachable -- two checks that a row is under a limit nothing can exceed would pass for free.
static const archs_table g_table_absurd = { 400, 400 };

static std::string build_row(const archs_table & table, const row_case & rc) {
    return table.row(rc.arch, rc.device, rc.config, rc.status_nmse, rc.nmse_value, rc.status_roundtrip);
}

// Emits one row too long for emit()'s bound, so the gate can watch the warning fire.
static void body_oversize() {
    archs_table::emit(build_row(g_table_absurd, g_rows[3]));
}

// The emission this ticket installed: the breadcrumb and the log noise go to stderr, the row is
// assembled whole and written after the work it describes.
static void body_fixed() {
    archs_table::emit(g_table.header(g_nmse_gate));
    for (size_t i = 0; i < g_n_rows; i++) {
        const row_case & rc = g_rows[i];
        fprintf(stderr, "test case: %s (%s, %s)\n", rc.arch, rc.device, rc.config);
        log_noise();
        archs_table::emit(build_row(g_table, rc));
    }
    archs_table::emit(g_table.footer(g_nmse_gate, 1));
}

// The emission this ticket removed, reproduced faithfully so the gate has a control it can
// observe firing: row prefix, fflush(stdout), the work (and its log output), then the result.
static void body_pre_fix() {
    archs_table::emit(g_table.header(g_nmse_gate));
    const std::string template_row_cfg =
        "|%" + std::to_string(g_table.width_arch) + "s|%" + std::to_string(g_table.width_device) + "s|%6s|";
    for (size_t i = 0; i < g_n_rows; i++) {
        const row_case & rc = g_rows[i];
        printf(template_row_cfg.c_str(), rc.arch, rc.device, rc.config);
        fflush(stdout);
        log_noise();
        printf("%15s %10s|%20s|\n", rc.status_nmse, rc.nmse_value, rc.status_roundtrip);
    }
    fflush(stdout);
}

// Runs `body` with stdout and stderr pointing at ONE file description -- exactly `> log 2>&1`,
// which is how the sweeps in artifacts/ were captured and where the corruption showed up.
// Returns false only if the platform would not give us a temp file (Windows without
// administrator privileges); the caller reports that as a skipped sub-check, not a pass.
static bool capture_merged(void (*body)(), std::string & out) {
    FILE * file = tmpfile();
    if (file == nullptr) {
        return false;
    }

    fflush(stdout);
    fflush(stderr);
    const int saved_stdout = archs_dup(archs_fileno(stdout));
    const int saved_stderr = archs_dup(archs_fileno(stderr));
    archs_dup2(archs_fileno(file), archs_fileno(stdout));
    archs_dup2(archs_fileno(file), archs_fileno(stderr));

    body();

    fflush(stdout);
    fflush(stderr);
    archs_dup2(saved_stdout, archs_fileno(stdout));
    archs_dup2(saved_stderr, archs_fileno(stderr));
    archs_close(saved_stdout);
    archs_close(saved_stderr);

    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    out.assign(size > 0 ? (size_t) size : 0, '\0');
    if (!out.empty()) {
        out.resize(fread(&out[0], 1, out.size(), file));
    }
    fclose(file);
    return true;
}

// Verbatim from artifacts/1p2k/1p2k-s7-a.txt, lines 4-9: two intact rows, then a row the
// backend's first log line cut in half. Real pre-fix output from a real sweep, kept here so the
// parser's positive control does not depend on that file surviving. One string literal per
// captured line, so that "verbatim" stays checkable by eye.
// clang-format off
static const char * g_historical_excerpt =
    "|            clip|Intel(R) Arc(TM) Pro B70 Graphics| Dense|\033[1;33mSKIP\033[0m           |     \033[1;33mSKIP\033[0m|\n"
    "|            clip|   Intel(R) Core(TM) Ultra 7 265K| Dense|\033[1;33mSKIP\033[0m           |     \033[1;33mSKIP\033[0m|\n"
    "|           llama|Intel(R) Arc(TM) Pro B70 Graphics| Dense|[SYCL] GGML_SYCL_F16 build: attention Q/accumulators are f16\n"
    "  \033[1;32mOK\033[0m (1.07e-10)|       \033[1;32mOK\033[0m|\n"
    "|           llama|   Intel(R) Core(TM) Ultra 7 265K| Dense|  \033[1;32mOK\033[0m (0.00e+00)|       \033[1;32mOK\033[0m|\n";
// clang-format on

static int check_file(const char * path) {
    FILE * file = fopen(path, "rb");
    if (file == nullptr) {
        fprintf(stderr, "%s: cannot open %s\n", __func__, path);
        return 1;
    }
    std::string text;
    char        buf[4096];
    size_t      n_read = 0;
    while ((n_read = fread(buf, 1, sizeof(buf), file)) > 0) {
        text.append(buf, n_read);
    }
    fclose(file);

    const parse_result res = parse_table(text, true);
    printf("%s: %zu table line(s): %zu well-formed, %zu mangled; %zu orphaned row tail(s)\n", path, res.n_table,
           res.n_well_formed, res.n_mangled, res.n_orphan_tail);
    if (res.n_table == 0) {
        // An empty result is not a clean one. A log with no table in it never ran the harness,
        // and reporting "0 mangled" for it would be the same fails-open answer this gate exists
        // to remove.
        printf("%s: no table found -- this proves NOTHING about that run\n", path);
        return 1;
    }
    return res.n_mangled == 0 && res.n_orphan_tail == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--check") == 0) {
        return check_file(argv[2]);
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--check <captured-sweep-log>]\n", argv[0]);
        return 1;
    }

    // 1. The emitter, under the interleaving that used to break it.
    std::string out_fixed;
    if (!capture_merged(body_fixed, out_fixed)) {
        fprintf(stderr, "SKIP: tmpfile() unavailable, cannot capture a merged stream\n");
        return 77;
    }
    const parse_result fixed = parse_table(out_fixed, true);

    // Assert the pressure was actually applied. Without this, a capture that lost the log output
    // would report a spotless table and prove nothing -- there would have been nothing to corrupt.
    check(out_fixed.find("[UNIFIED-CACHE]") != std::string::npos,
          "log output really is interleaved into the captured stream");
    check(out_fixed.find("test case: gemma2 ") != std::string::npos, "the crash breadcrumb survives, on stderr");
    check(fixed.n_table == g_n_rows + 2, "header, separator and every row reach the table");
    check(fixed.n_mangled == 0, "no mangled table line under interleaved log output");
    check(fixed.n_orphan_tail == 0, "no orphaned row tail under interleaved log output");
    check(n_well_formed_with(out_fixed, "FAIL") == 1, "the FAIL row parses -- `grep FAIL` can find it");

    // 2. The legend and the footer are not rows, and a strict parser must not count them as
    //    malformed ones. This is what lets the table carry prose at all.
    const parse_result legend = parse_table(g_table.header(g_nmse_gate), false);
    const parse_result footer = parse_table(g_table.footer(g_nmse_gate, 0), false);
    check(legend.n_table == 2 && legend.n_mangled == 0, "legend lines are not parsed as rows");
    check(footer.n_table == 0 && footer.n_orphan_tail == 0, "the closing gate line is not parsed as a row");

    // 3. Positive control: the emission this ticket replaced, through the same parser. If this
    //    comes back clean the gate above is incapable of failing and means nothing.
    std::string out_pre_fix;
    if (!capture_merged(body_pre_fix, out_pre_fix)) {
        fprintf(stderr, "SKIP: tmpfile() unavailable, cannot capture a merged stream\n");
        return 77;
    }
    const parse_result pre_fix = parse_table(out_pre_fix, false);
    check(pre_fix.n_mangled == g_n_rows, "control: pre-fix emission mangles every row");
    check(pre_fix.n_orphan_tail == g_n_rows, "control: pre-fix emission orphans every row tail");
    check(n_well_formed_with(out_pre_fix, "FAIL") == 0,
          "control: pre-fix, `grep FAIL` over well-formed rows finds nothing on a failing run");

    // 4. emit()'s size precondition, pinned rather than assumed. The comment on
    //    archs_table::max_atomic_line does arithmetic; this makes the arithmetic fail loudly if a
    //    future column, a wider status string or a longer device description eats the margin.
    const row_case & widest = g_rows[3];  // BITDIFF -- the longest roundtrip status
    check(build_row(g_table, widest).size() < archs_table::max_atomic_line,
          "a real-width row fits emit()'s atomic-write bound");
    check(build_row(g_table_wide, widest).size() < archs_table::max_atomic_line,
          "so does a row far wider than any device this fork has seen");
    check(build_row(g_table_absurd, widest).size() >= archs_table::max_atomic_line,
          "control: an absurd width DOES exceed the bound, so it is not vacuously large");

    // And the warning must actually fire, or the runtime check is decorative. Captured, because
    // it goes to stderr by design -- a warning printed into the table would be the defect itself.
    std::string out_oversize;
    if (!capture_merged(body_oversize, out_oversize)) {
        fprintf(stderr, "SKIP: tmpfile() unavailable, cannot capture a merged stream\n");
        return 77;
    }
    check(out_oversize.find("exceeds the 256-byte atomic-write bound") != std::string::npos,
          "control: emit() warns when a line exceeds the bound");
    check(parse_table(out_oversize, false).n_table == 1, "the oversized line is still emitted, not dropped");

    // 5. Positive control on real pre-fix output rather than a reproduction of it.
    const parse_result historical = parse_table(g_historical_excerpt, false);
    check(historical.n_well_formed == 3, "historical excerpt: the intact rows still parse");
    check(historical.n_mangled == 1, "control: historical excerpt has a mangled row");
    check(historical.n_orphan_tail == 1, "control: historical excerpt has an orphaned row tail");

    printf("%s: %d check(s) failed\n", argv[0], g_n_fail);
    return g_n_fail == 0 ? 0 : 1;
}
